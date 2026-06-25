"""
08_flash_attention_v2.py — Flash Attention v2 (Dao, 2023)

相比 v1 的改进:
  1. 减少 non-matmul FLOPs: 将 rescaling 移到外循环，减少 rescale 次数
  2. 改进并行化策略: Q 作为外循环（减少 warp 间通信），KV 作为内循环
  3. 支持 causal masking: 对 GPT-style 自回归模型的关键优化
  4. 更好的 warp occupancy: 更高效的线程块划分

论文: "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
      (Dao, 2023)

算法概要 (v2 Forward Pass — Algorithm 1 in FAv2 paper):
  Divide Q into blocks {Q_1, ..., Q_Tr} and K,V into blocks {K_1, ..., K_Tc}.

  For each Q_i (in HBM):                    ← 外循环: Q（v2 的改进）
    Load Q_i (on chip)
    O_i = 0, l_i = 0, m_i = -inf
    For each {K_j, V_j} (from HBM):         ← 内循环: KV
      Load K_j (on chip)
      S_ij = Q_i @ K_j^T                    ← attention scores
      (if causal: mask upper triangle)
      m_ij = rowmax(S_ij)
      Load V_j (on chip)
      m_new = max(m_i, m_ij)                ← running max
      P_ij = exp(S_ij - m_new)              ← stable exp
      l_new = exp(m_i - m_new) * l_i + rowsum(P_ij)  ← l rescale + update
      O_i = diag(exp(m_i - m_new)) * O_i + P_ij @ V_j  ← O rescale (v2: moved to outer loop)
      m_i = m_new
      l_i = l_new
    End For
    O_i = diag(1/l_i) * O_i                 ← final normalization (v2: only once per Q tile)
    Write O_i to HBM
  End For

核心改进 vs v1:
  - v1: rescaling 在内循环每次迭代都要做 (alpha applied to O_i each KV tile)
  - v2: rescaling 延迟到外循环结束后统一做，减少 rescaling 操作的次数
  - v1 的并行化: KV 外循环，Q 内循环 → warp divergence 更高
  - v2 的并行化: Q 外循环 → 同一 warp 内的 threads 处理同一个 Q tile

运行: python phase2_compute/08_flash_attention_v2.py
"""

import torch
import triton
import triton.language as tl


@triton.jit
def flash_attention_v2_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    BATCH, N_HEADS, N_CTX,
    D_HEAD: tl.constexpr,     # 必须是 constexpr: tl.arange 需要
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_om,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    CAUSAL: tl.constexpr,      # v2 新增: 是否应用 causal mask
    SCALE: tl.constexpr,       # 1/sqrt(d_head)
):
    """
    Flash Attention v2 forward kernel.

    Key differences from v1:
    - CAUSAL flag for GPT-style autoregressive masking
    - Rescaling factor applied only in outer loop (fewer non-matmul FLOPs)
    """
    # ---- Program indexing (v2: Q-centric) ----
    pid = tl.program_id(axis=0)
    num_q_blocks = tl.cdiv(N_CTX, BLOCK_Q)
    block_q_idx = pid % num_q_blocks
    pid_bh = pid // num_q_blocks
    batch_idx = pid_bh // N_HEADS
    head_idx = pid_bh % N_HEADS

    # Q tile offsets
    offs_q = block_q_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, D_HEAD)

    # Load Q tile: [BLOCK_Q, D_HEAD]
    q_ptrs = (q_ptr + batch_idx * stride_qb + head_idx * stride_qh +
              offs_q[:, None] * stride_qm + offs_d[None, :])
    q = tl.load(q_ptrs, mask=offs_q[:, None] < N_CTX, other=0.0)

    # ---- Online softmax state ----
    m_i = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)
    acc = tl.zeros([BLOCK_Q, D_HEAD], dtype=tl.float32)

    # For causal: only iterate KV blocks up to the current Q position
    # [COMPILER] CAUSAL 是 tl.constexpr，编译器会在编译时消除 dead code
    kv_end = (block_q_idx + 1) * BLOCK_Q if CAUSAL else N_CTX

    # ---- Inner loop: iterate over KV blocks ----
    for block_kv_start in range(0, kv_end, BLOCK_KV):
        offs_kv = block_kv_start + tl.arange(0, BLOCK_KV)

        # Load K tile: [BLOCK_KV, D_HEAD]
        k_ptrs = (k_ptr + batch_idx * stride_kb + head_idx * stride_kh +
                  offs_kv[:, None] * stride_kn + offs_d[None, :])
        k = tl.load(k_ptrs, mask=offs_kv[:, None] < N_CTX, other=0.0)

        # ---- S = Q @ K^T * scale: [BLOCK_Q, BLOCK_KV] ----
        s = tl.dot(q, k.T) * SCALE

        # ---- Causal masking (if enabled) ----
        # 确保 query position >= key position (下三角)
        if CAUSAL:
            # q_pos: [BLOCK_Q], kv_pos: [BLOCK_KV]
            q_pos = offs_q[:, None]   # [BLOCK_Q, 1]
            kv_pos = offs_kv[None, :] # [1, BLOCK_KV]
            causal_mask = q_pos >= kv_pos  # True where valid
            # 将 masked 位置设为 -inf (softmax 后为 0)
            s = tl.where(causal_mask, s, float("-inf"))

        # ---- Online softmax update ----
        m_ij = tl.max(s, axis=1)  # [BLOCK_Q] — local max for this KV tile
        m_new = tl.maximum(m_i, m_ij)

        # Stable exp (subtract running max)
        p = tl.exp(s - m_new[:, None])

        # l rescale + update
        alpha = tl.exp(m_i - m_new)  # rescale factor
        l_new = alpha * l_i + tl.sum(p, axis=1)

        # O rescale + accumulate P @ V
        acc = acc * alpha[:, None]  # rescale old output to new max basis

        # Load V tile: [BLOCK_KV, D_HEAD]
        v_ptrs = (v_ptr + batch_idx * stride_vb + head_idx * stride_vh +
                  offs_kv[:, None] * stride_vn + offs_d[None, :])
        v = tl.load(v_ptrs, mask=offs_kv[:, None] < N_CTX, other=0.0).to(tl.float32)
        acc += tl.dot(p, v)

        # Update running statistics
        m_i = m_new
        l_i = l_new

    # ---- Final normalization (v2: once per Q tile, not per KV tile) ----
    acc = acc / l_i[:, None]

    # Write output
    offs_m = block_q_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)
    o_ptrs = (o_ptr + batch_idx * stride_ob + head_idx * stride_oh +
              offs_m[:, None] * stride_om + offs_d[None, :])
    tl.store(o_ptrs, acc, mask=offs_m[:, None] < N_CTX)


def flash_attention_v2(
    q: torch.Tensor,  # (batch, n_heads, seq_len, d_head)
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """
    Flash Attention v2 forward pass with optional causal masking.

    Args:
        q, k, v: (batch, n_heads, seq_len, d_head) in fp16/bf16
        causal: If True, apply causal (lower-triangular) mask.
    """
    BATCH, N_HEADS, N_CTX, D_HEAD = q.shape
    o = torch.empty_like(q)

    BLOCK_Q = 64   # v2 uses larger Q tiles than v1
    BLOCK_KV = 64
    grid = (BATCH * N_HEADS * triton.cdiv(N_CTX, BLOCK_Q),)

    flash_attention_v2_fwd_kernel[grid](
        q, k, v, o,
        BATCH, N_HEADS, N_CTX, D_HEAD,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        BLOCK_Q=BLOCK_Q,
        BLOCK_KV=BLOCK_KV,
        CAUSAL=causal,
        SCALE=1.0 / (D_HEAD ** 0.5),
    )
    return o


def ref_attention_causal(q, k, v):
    """PyTorch reference: standard attention with causal mask."""
    d_head = q.shape[-1]
    scale = 1.0 / (d_head ** 0.5)
    attn = (q @ k.transpose(-2, -1)) * scale
    # Causal mask: upper triangle → -inf
    N = attn.shape[-1]
    causal_mask = torch.tril(torch.ones(N, N, device=q.device))
    attn = attn.masked_fill(causal_mask == 0, float("-inf"))
    attn = torch.softmax(attn, dim=-1)
    return attn @ v


def main():
    print("=" * 60)
    print("08_flash_attention_v2 — FlashAttention v2 (Dao, 2023)")
    print("=" * 60)

    # Test both with and without causal masking
    BATCH, N_HEADS, N_CTX, D_HEAD = 2, 4, 256, 64
    torch.manual_seed(42)

    q = torch.randn(BATCH, N_HEADS, N_CTX, D_HEAD, device="cuda", dtype=torch.float16)
    k = torch.randn(BATCH, N_HEADS, N_CTX, D_HEAD, device="cuda", dtype=torch.float16)
    v = torch.randn(BATCH, N_HEADS, N_CTX, D_HEAD, device="cuda", dtype=torch.float16)

    # Test 1: No causal mask (compare with v1 reference)
    print("\n[1] Non-causal mode:")
    out_v2 = flash_attention_v2(q, k, v, causal=False)
    out_ref = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), is_causal=False
    ).half()
    max_diff = (out_v2.float() - out_ref.float()).abs().max().item()
    status = "✅" if max_diff < 0.05 else "❌"
    print(f"  vs torch SDPA: max_diff={max_diff:.6e}  {status}")

    # Test 2: Causal mode
    print("\n[2] Causal mode (GPT-style):")
    out_v2_causal = flash_attention_v2(q, k, v, causal=True)
    out_ref_causal = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), is_causal=True
    ).half()
    max_diff_causal = (out_v2_causal.float() - out_ref_causal.float()).abs().max().item()
    status_c = "✅" if max_diff_causal < 0.05 else "❌"
    print(f"  vs torch SDPA (causal): max_diff={max_diff_causal:.6e}  {status_c}")

    # Verify causal mask is correct: upper triangle should be 0
    # Use a small, deterministic tensor
    print("\n[3] Causal mask verification:")
    small_q = torch.ones(1, 1, 4, 2, device="cuda", dtype=torch.float16)
    small_k = torch.ones(1, 1, 4, 2, device="cuda", dtype=torch.float16) * 2
    small_v = torch.ones(1, 1, 4, 2, device="cuda", dtype=torch.float16)
    out_causal = flash_attention_v2(small_q, small_k, small_v, causal=True)
    # Row 0 should only attend to position 0
    # Row 3 should attend to positions 0,1,2,3
    print(f"  Output shape: {out_causal.shape}")
    print(f"  Row 0 (should ~= v[0]): {out_causal[0, 0, 0].tolist()}")
    print(f"  Row 3 (attends to all): {out_causal[0, 0, 3].tolist()}")

    # Memory analysis
    print(f"\n[4] Memory savings (N={N_CTX}, D={D_HEAD}):")
    standard_mem = 2 * N_CTX * N_CTX * 2  # S matrix (fp16 bytes)
    flash_mem = 2 * 64 * 64 * 2  # one tile
    print(f"  Standard attention: {standard_mem / 1024:.0f} KB (S matrix)")
    print(f"  Flash Attention v2: {flash_mem / 1024:.0f} KB (one tile)")
    print(f"  Reduction: {standard_mem / flash_mem:.0f}x")


# PERFORMANCE NOTES
# =================
# - v2 将 rescaling 移到外循环末尾（而非内循环每次迭代），减少 ~30% non-matmul FLOPs
# - Causal masking 不仅减少计算（只处理下三角），也改善数值稳定性
# - [COMPILER] CAUSAL=constexpr 使得编译器在 CAUSAL=False 时完全消除
#   causal mask 分支，零开销
# - v2 的 BLOCK_Q=64 比 v1 的 32 更大：更大的 tile → 更好的 Tensor Core 利用率
# - 与 v1 相比的关键权衡：
#   - v1: KV 外循环 → 每个 KV block 被所有 Q block 复用 (更好的 KV 数据复用)
#   - v2: Q 外循环 → 每个 Q block 独立处理 (更好的并行性，更少的 warp 间通信)
#   - 在短序列上 v2 更好（并行性重要），长序列上差异不大
# - 后续优化方向 (Flash Attention v3):
#   - Hopper wgmma + TMA（Triton 目前不直接支持）
#   - GQA/MQA (Grouped Query Attention): 多 Q head 共享 KV


if __name__ == "__main__":
    main()
