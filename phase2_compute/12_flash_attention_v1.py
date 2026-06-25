"""
07_flash_attention_v1.py — Flash Attention v1 (Dao et al., 2022)

学习目标：
  - 理解 Flash Attention 的核心算法：online softmax + tiling
  - 掌握 O(N²) → O(N) SRAM 的优化原理
  - 学会 block-by-block softmax 的数学推导

论文: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
      (Dao, Fu, Ermon, Rudra, Ré; NeurIPS 2022)

算法概要 (Algorithm 1 — Forward Pass):
  Divide Q into blocks {Q_1, ..., Q_Tr} and K,V into blocks {K_1, ..., K_Tc}.
  For each Q_i (in HBM):
    Load Q_i, O_i = 0, l_i = 0, m_i = -inf (on chip)
    For each {K_j, V_j} (from HBM):
      Load K_j, V_j (on chip)
      S_ij = Q_i @ K_j^T                          ← attention scores
      m_ij = rowmax(S_ij)                          ← local max
      m_new = max(m_i, m_ij)                       ← running max
      P_ij = exp(S_ij - m_new)                     ← stable exp
      l_new = exp(m_i - m_new) * l_i + rowsum(P_ij) ← running sum (online softmax)
      O_i = diag(exp(m_i - m_new)) * O_i + P_ij @ V_j  ← weighted sum
      m_i = m_new, l_i = l_new                     ← update running statistics
    End For
    O_i = diag(1/l_i) * O_i                        ← final normalization
    Write O_i to HBM
  End For

运行: python phase2_compute/07_flash_attention_v1.py
"""

import torch
import triton
import triton.language as tl


@triton.jit
def flash_attention_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    BATCH, N_HEADS, N_CTX,
    stride_qb, stride_qh, stride_qm,  # Q strides
    stride_kb, stride_kh, stride_kn,  # K strides
    stride_vb, stride_vh, stride_vn,  # V strides
    stride_ob, stride_oh, stride_om,  # O strides
    BLOCK_Q: tl.constexpr,  # Q tile size (在序列维)
    BLOCK_KV: tl.constexpr,  # KV tile size (在序列维)
    D_HEAD: tl.constexpr,  # [COMPILER] must be constexpr for tl.arange
    SCALE: tl.constexpr,   # 1/sqrt(d_head) softmax scaling
):
    """
    Flash Attention v1 forward kernel.

    每个 program 负责计算一个 (batch, head) 对的一个 Q tile 的输出。
    """
    # 当前 program 负责的 (batch, head, Q tile)
    pid = tl.program_id(axis=0)
    block_q_idx = pid % (N_CTX // BLOCK_Q)  # 简化: 假设 N_CTX 被 BLOCK_Q 整除
    pid_bh = pid // (N_CTX // BLOCK_Q)
    batch_idx = pid_bh // N_HEADS
    head_idx = pid_bh % N_HEADS

    # Q tile 的序列维偏移
    offs_q = block_q_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, D_HEAD)

    # 加载 Q tile: [BLOCK_Q, D_HEAD]
    q_ptrs = (q_ptr + batch_idx * stride_qb + head_idx * stride_qh +
              offs_q[:, None] * stride_qm + offs_d[None, :])
    q = tl.load(q_ptrs, mask=offs_q[:, None] < N_CTX, other=0.0)

    # ---- Online softmax 状态 (Algorithm 1 的 l, m, O) ----
    m_i = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)  # running max
    l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)                 # running sum
    acc = tl.zeros([BLOCK_Q, D_HEAD], dtype=tl.float32)         # running output

    # ---- 沿 KV 序列维迭代 ----
    for block_kv_start in range(0, N_CTX, BLOCK_KV):
        offs_kv = block_kv_start + tl.arange(0, BLOCK_KV)

        # Load K tile: [BLOCK_KV, D_HEAD]
        k_ptrs = (k_ptr + batch_idx * stride_kb + head_idx * stride_kh +
                  offs_kv[:, None] * stride_kn + offs_d[None, :])
        k = tl.load(k_ptrs, mask=offs_kv[:, None] < N_CTX, other=0.0)

        # ---- S = Q @ K^T / sqrt(d_head): [BLOCK_Q, BLOCK_KV] ----
        # 缩放因子 1/sqrt(d_k) 对注意力分数的分布至关重要
        s = tl.dot(q, k.T) * SCALE

        # ---- Online softmax update ----
        # m_ij = rowmax(s)  — local max for this KV tile
        m_ij = tl.max(s, axis=1)  # [BLOCK_Q]

        # m_new = max(m_i, m_ij)  — running max across all KV tiles
        m_new = tl.maximum(m_i, m_ij)

        # P_ij = exp(S - m_new)  — stable softmax numerator
        p = tl.exp(s - m_new[:, None])

        # l_new = exp(m_i - m_new) * l_i + rowsum(P_ij)
        # 修正因子: 将旧的 sum 缩放到新的 max 基准
        alpha = tl.exp(m_i - m_new)  # [BLOCK_Q]
        l_new = alpha * l_i + tl.sum(p, axis=1)

        # O_i = diag(alpha) * O_i + P_ij @ V_j
        # 旧输出需要缩放到新的 max 基准，然后加上当前贡献
        acc = acc * alpha[:, None]  # rescale old output
        v_ptrs = (v_ptr + batch_idx * stride_vb + head_idx * stride_vh +
                  offs_kv[:, None] * stride_vn + offs_d[None, :])
        v = tl.load(v_ptrs, mask=offs_kv[:, None] < N_CTX, other=0.0).to(tl.float32)
        acc += tl.dot(p, v)  # P @ V

        # 更新 running statistics
        m_i = m_new
        l_i = l_new

    # ---- 最终归一化 ----
    # O = diag(1/l_i) * O_i
    acc = acc / l_i[:, None]

    # Write output
    offs_m = block_q_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)
    o_ptrs = (o_ptr + batch_idx * stride_ob + head_idx * stride_oh +
              offs_m[:, None] * stride_om + offs_d[None, :])
    tl.store(o_ptrs, acc, mask=offs_m[:, None] < N_CTX)


def flash_attention_v1(
    q: torch.Tensor,  # (batch, n_heads, seq_len, d_head)
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """
    Flash Attention v1 forward pass.
    """
    BATCH, N_HEADS, N_CTX, D_HEAD = q.shape
    o = torch.empty_like(q)

    BLOCK_Q = 32
    BLOCK_KV = 32
    grid = (BATCH * N_HEADS * triton.cdiv(N_CTX, BLOCK_Q),)

    flash_attention_fwd_kernel[grid](
        q, k, v, o,
        BATCH, N_HEADS, N_CTX,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        BLOCK_Q=BLOCK_Q,
        BLOCK_KV=BLOCK_KV,
        D_HEAD=D_HEAD,
        SCALE=1.0 / (D_HEAD ** 0.5),
    )
    return o


def ref_attention(q, k, v):
    """PyTorch reference: standard scaled dot-product attention."""
    d_head = q.shape[-1]
    scale = 1.0 / (d_head ** 0.5)
    attn = (q @ k.transpose(-2, -1)) * scale
    attn = torch.softmax(attn, dim=-1)
    return attn @ v


def main():
    print("=" * 60)
    print("07_flash_attention_v1 — FlashAttention (Dao et al., 2022)")
    print("=" * 60)

    # 小规模测试
    BATCH, N_HEADS, N_CTX, D_HEAD = 2, 4, 256, 64
    torch.manual_seed(42)

    q = torch.randn(BATCH, N_HEADS, N_CTX, D_HEAD, device="cuda", dtype=torch.float16)
    k = torch.randn(BATCH, N_HEADS, N_CTX, D_HEAD, device="cuda", dtype=torch.float16)
    v = torch.randn(BATCH, N_HEADS, N_CTX, D_HEAD, device="cuda", dtype=torch.float16)

    # 正确性验证
    out_triton = flash_attention_v1(q, k, v)
    out_ref = ref_attention(q.float(), k.float(), v.float()).half()

    max_diff = (out_triton.float() - out_ref.float()).abs().max().item()
    print(f"  Shape: ({BATCH}, {N_HEADS}, {N_CTX}, {D_HEAD})")
    print(f"  Max diff: {max_diff:.6e}  {'✅' if max_diff < 0.01 else '❌'}")

    # 内存分析
    standard_mem = 2 * N_CTX * N_CTX * 2  # attention matrix (fp16, bytes)
    flash_mem = 2 * 32 * 32 * 2  # one tile (bytes)
    print(f"\n  Memory (attention matrix):")
    print(f"    Standard:  {standard_mem / 1024:.0f} KB")
    print(f"    Flash v1:  {flash_mem / 1024:.0f} KB (per tile)")
    print(f"    Reduction: {standard_mem / flash_mem:.0f}x")


# PERFORMANCE NOTES
# =================
# - 核心洞察: attention matrix (N²) 不需要完全物化到 HBM
# - Online softmax (Milakov & Gimelshein, 2018):
#   在分块计算 softmax 时，用 running max/sum 维护全局统计量
#   每个新 block 到来时，用 rescaling 因子修正旧值
# - [COMPILER] Online softmax 是 compiler-friendly 的算法:
#   - 所有中间量 (m, l, acc) 都在寄存器中
#   - 没有跨程序的依赖（每个 program 独立处理自己的 Q tile）
# - 内存分析:
#   - Standard: O(N²) HBM → 对于 seq_len=4096, ~32MB per head
#   - Flash: O(sqrt(N)) SRAM → 只需缓存一个 tile
# - 本实现是简化的 v1，缺少:
#   - Softmax scaling (1/sqrt(d))
#   - Causal masking
#   - Backward pass (TODO: 单独文件)


if __name__ == "__main__":
    main()
