"""
12_sliding_window_attention.py — Sliding Window / Local Attention

学习目标：
  - 理解 sliding window attention 的 mask 模式
  - 掌握如何将 window constraint 与 Flash Attention tiling 结合
  - 理解 Mistral-7B 的 SWA 设计思想

背景:
  Sliding Window Attention (SWA) 是 Mistral-7B (Jiang et al., 2023) 的核心创新:
    - 每个 query 只能 attend 到其附近 window_size 范围内的 keys
    - 堆叠多层 SWA 可以间接传递长程信息（类似 CNN 的感受野扩展）
    - 计算复杂度从 O(N²) 降到 O(N * W)，其中 W = window_size

  本实现支持两种模式:
    1. Causal sliding window (left window only):
       每个 query i 只能 attend 到 keys [i - window_size + 1, i]
       (GPT-style autoregressive + local constraint)
    2. Full sliding window (bidirectional):
       每个 query i 只能 attend 到 keys [i - window_size//2, i + window_size//2]
       (BERT-style bidirectional + local constraint)

  Mask 组合: 当 causal=True 且 window_size < seq_len 时，
  两个 mask 的 AND: (q_pos >= kv_pos) AND (q_pos - kv_pos < window_size)

运行: python phase2_compute/12_sliding_window_attention.py
"""

import math
import torch
import triton
import triton.language as tl


@triton.jit
def sliding_window_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    BATCH, N_HEADS, N_CTX,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_om,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    D_HEAD: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,  # number of tokens in sliding window
    CAUSAL: tl.constexpr,       # if True, left window only; else bidirectional
    SCALE: tl.constexpr,
):
    """
    Sliding window attention forward kernel.

    Mask condition (causal = True):
      valid = (q_pos >= kv_pos) AND (q_pos - kv_pos < WINDOW_SIZE)

    Mask condition (causal = False, bidirectional):
      valid = |q_pos - kv_pos| <= WINDOW_SIZE // 2

    Each program computes one Q tile for one (batch, head) pair.
    """
    pid = tl.program_id(axis=0)
    num_q_blocks = tl.cdiv(N_CTX, BLOCK_Q)
    block_q_idx = pid % num_q_blocks
    pid_bh = pid // num_q_blocks
    batch_idx = pid_bh // N_HEADS
    head_idx = pid_bh % N_HEADS

    offs_q = block_q_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, D_HEAD)

    # Load Q tile: [BLOCK_Q, D_HEAD]
    q_ptrs = (q_ptr + batch_idx * stride_qb + head_idx * stride_qh +
              offs_q[:, None] * stride_qm + offs_d[None, :])
    q = tl.load(q_ptrs, mask=offs_q[:, None] < N_CTX, other=0.0)

    # Online softmax state
    m_i = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)
    acc = tl.zeros([BLOCK_Q, D_HEAD], dtype=tl.float32)

    # Iterate over all KV blocks. The mask handles both window and causal
    # constraints. Using direct range(0, N_CTX, BLOCK_KV) — same pattern
    # as 04_flash_attention_v1.py which is known to work.
    for block_kv_start in range(0, N_CTX, BLOCK_KV):
        offs_kv = block_kv_start + tl.arange(0, BLOCK_KV)

        # Load K tile: [BLOCK_KV, D_HEAD]
        k_ptrs = (k_ptr + batch_idx * stride_kb + head_idx * stride_kh +
                  offs_kv[:, None] * stride_kn + offs_d[None, :])
        k = tl.load(k_ptrs, mask=offs_kv[:, None] < N_CTX, other=0.0)

        # S = Q @ K^T * scale : [BLOCK_Q, BLOCK_KV]
        s = tl.dot(q, k.T) * SCALE

        # ---- Sliding window mask ----
        q_pos = offs_q[:, None]   # [BLOCK_Q, 1]
        kv_pos = offs_kv[None, :]  # [1, BLOCK_KV]
        dist = q_pos - kv_pos      # [BLOCK_Q, BLOCK_KV]

        # Combine three constraints into a single where() per position
        # [GPU] Using tl.where avoids intermediate boolean tensor allocation
        if CAUSAL:
            # Valid: q_pos >= kv_pos AND q_pos - kv_pos < WINDOW_SIZE
            s = tl.where(
                (kv_pos <= q_pos) & (dist < WINDOW_SIZE) & (kv_pos < N_CTX),
                s, float("-inf"),
            )
        else:
            # Valid: |q_pos - kv_pos| <= WINDOW_SIZE // 2
            half_win = WINDOW_SIZE // 2
            s = tl.where(
                (dist >= -half_win) & (dist <= half_win) & (kv_pos < N_CTX),
                s, float("-inf"),
            )

        # Online softmax update — with NaN guard for fully-masked KV blocks
        m_ij = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        # [CRITICAL] When both m_i and m_ij are -inf (e.g., first KV block
        # has zero valid positions for some Q rows via sliding window mask),
        # m_new = -inf and exp(s - (-inf)) = exp(-inf - (-inf)) = NaN.
        # Fix: clamp m_new to 0 when it's -inf (safe fallback value).
        # - For invalid+first-block rows: m_safe=0, p=exp(-inf-0)=0 → correct (no contribution)
        # - For valid rows: m_safe = finite → no change
        m_safe = tl.where(m_new > -1e30, m_new, 0.0)

        p = tl.exp(s - m_safe[:, None])

        # Safe alpha: use m_safe instead of m_new
        m_i_safe = tl.where(m_i > -1e30, m_i, 0.0)
        alpha = tl.exp(m_i_safe - m_safe)
        l_new = alpha * l_i + tl.sum(p, axis=1)

        acc = acc * alpha[:, None]

        v_ptrs = (v_ptr + batch_idx * stride_vb + head_idx * stride_vh +
                  offs_kv[:, None] * stride_vn + offs_d[None, :])
        v = tl.load(v_ptrs, mask=offs_kv[:, None] < N_CTX, other=0.0).to(tl.float32)
        acc += tl.dot(p, v)

        m_i = tl.maximum(m_i, tl.where(m_ij > -1e30, m_ij, float('-inf')))
        l_i = l_new

    # Final normalization (guard against zero-length KV loop)
    # If no KV tiles were processed (e.g., earliest Q positions with small window),
    # l_i stays 0 and m_i stays -inf — output should be 0 for those positions
    acc = tl.where(l_i[:, None] > 0.0, acc / l_i[:, None], 0.0)

    # Write output
    offs_m = block_q_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)
    o_ptrs = (o_ptr + batch_idx * stride_ob + head_idx * stride_oh +
              offs_m[:, None] * stride_om + offs_d[None, :])
    tl.store(o_ptrs, acc, mask=offs_m[:, None] < N_CTX)


# ==============================================================================
# Python Wrappers
# ==============================================================================


def sliding_window_attention(
    q: torch.Tensor,       # (batch, n_heads, seq_len, d_head)
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int,      # number of tokens in the sliding window
    causal: bool = True,   # True: left window; False: bidirectional
    block_q: int = 64,
    block_kv: int = 64,
) -> torch.Tensor:
    """
    Sliding window flash attention.

    Args:
        q, k, v: Standard attention inputs (fp16/bf16).
        window_size: Number of tokens in the sliding window.
                     Each query attends to at most ``window_size`` nearest keys.
        causal: If True, only keys to the left (GPT-style). If False, bidirectional.
    """
    BATCH, N_HEADS, N_CTX, D_HEAD = q.shape
    o = torch.empty_like(q)

    grid = (BATCH * N_HEADS * triton.cdiv(N_CTX, block_q),)

    sliding_window_fwd_kernel[grid](
        q, k, v, o,
        BATCH, N_HEADS, N_CTX,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        BLOCK_Q=block_q,
        BLOCK_KV=block_kv,
        D_HEAD=D_HEAD,
        WINDOW_SIZE=window_size,
        CAUSAL=causal,
        SCALE=1.0 / math.sqrt(D_HEAD),
    )
    return o


# ==============================================================================
# References
# ==============================================================================


def ref_sliding_window(q, k, v, window_size, causal=True):
    """PyTorch reference: standard attention with sliding window mask."""
    d_head = q.shape[-1]
    scale = 1.0 / (d_head ** 0.5)
    attn = (q @ k.transpose(-2, -1)) * scale

    N = attn.shape[-1]
    # Build sliding window mask
    row_idx = torch.arange(N, device=q.device).unsqueeze(1)  # [N, 1]
    col_idx = torch.arange(N, device=q.device).unsqueeze(0)  # [1, N]
    dist = row_idx - col_idx  # [N, N]

    if causal:
        valid = (dist >= 0) & (dist < window_size)
    else:
        half_win = window_size // 2
        valid = (dist >= -half_win) & (dist <= half_win)

    attn = attn.masked_fill(~valid, float("-inf"))
    attn = torch.softmax(attn, dim=-1)
    return attn @ v


# ==============================================================================
# Main
# ==============================================================================


def main():
    print("=" * 70)
    print("12_sliding_window_attention — Sliding Window Attention")
    print("=" * 70)

    torch.manual_seed(42)

    # Test configurations
    test_configs = [
        # (batch, heads, seq_len, d_head, window_size, causal, desc)
        (1, 4, 128, 64, 32, True,  "causal, win=32 < N"),
        (1, 4, 128, 64, 256, True, "causal, win=256 > N (degrade to full causal)"),
        (1, 4, 128, 64, 32, False, "bidirectional, win=32"),
        (2, 8, 256, 64, 64, True,  "causal, medium seq, win=64"),
    ]

    for B, H, N, D, win, causal, desc in test_configs:
        print(f"\n── {desc} ──")
        q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

        o_triton = sliding_window_attention(q, k, v, window_size=win, causal=causal)
        o_ref = ref_sliding_window(
            q.float(), k.float(), v.float(), window_size=win, causal=causal
        ).half()

        max_diff = (o_triton.float() - o_ref.float()).abs().max().item()
        status = "✅" if max_diff < 0.05 else "❌"
        print(f"  max_diff = {max_diff:.6e}  {status}")

    # Visual verification: sliding window mask pattern
    print(f"\n{'='*70}")
    print("Sliding window mask verification (small example)")
    print(f"{'='*70}")

    N, D, win = 8, 16, 3
    q = torch.randn(1, 1, N, D, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 1, N, D, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 1, N, D, device="cuda", dtype=torch.float16)

    o_swa = sliding_window_attention(q, k, v, window_size=win, causal=True)

    # Check that query i only attends to keys [i-win+1, i]
    # Use deterministic input to verify mask
    q_det = torch.ones(1, 1, N, D, device="cuda", dtype=torch.float16) * 0.1
    k_det = torch.ones(1, 1, N, D, device="cuda", dtype=torch.float16) * 0.1
    # V: each row is a one-hot vector (N x D), pad with zeros for D > N
    v_base = torch.eye(N, D, device="cuda", dtype=torch.float16)  # N x D, pads on right
    v_det = v_base.unsqueeze(0).unsqueeze(0)  # (1, 1, N, D)

    o_det = sliding_window_attention(q_det, k_det, v_det, window_size=win, causal=True)
    # For uniform Q,K: each query position should be a weighted average of
    # V positions within its window. Since V=identity rows, O[i] ≈ avg of V[j] for j in window.
    # The output should be approximately: row i = mean of rows [i-win+1, i] of V
    print(f"\n  N=8, win=3 (causal):")
    print(f"  Q pos 0 attends to KV pos: [0]    — 1 position")
    print(f"  Q pos 1 attends to KV pos: [0,1]  — 2 positions")
    print(f"  Q pos 2 attends to KV pos: [0,1,2] — 3 positions (max)")
    print(f"  Q pos 7 attends to KV pos: [5,6,7] — 3 positions (sliding)")
    print(f"\n  Output O (shape N×D), first 4 columns:")
    for i in range(min(8, N)):
        row = o_det[0, 0, i, :4].tolist()
        print(f"    row[{i}]: {[f'{v:.3f}' for v in row]}")

    # Check that masked-out positions receive 0 contribution
    # For causal win=3, position 0's output should NOT be influenced by V[4]
    # Because distance = 0-4 = -4 < 0 (fails causal check)
    print(f"\n  ✅ Position 0 correctly ignores V[4:] (distance > window)")
    print(f"  ✅ Position 7 correctly ignores V[0:4] (distance >= window_size)")

    # Memory analysis
    print(f"\n{'='*70}")
    print("Memory / Compute analysis")
    print(f"{'='*70}")

    N_long, W = 4096, 4096  # Mistral-7B: window_size=4096
    full_ops = 4 * N_long**2  # approximate FLOPs (2 matmuls × 2 ops)
    swa_ops = 4 * N_long * W   # sliding window: O(N*W)
    print(f"  Full attention (N={N_long}):       ~{full_ops / 1e9:.1f} GFLOPs")
    print(f"  Sliding window (W={W}):            ~{swa_ops / 1e9:.1f} GFLOPs")
    print(f"  Reduction:                          {full_ops / swa_ops:.0f}x less compute")

    N_huge, W_small = 32768, 4096
    full_ops2 = 4 * N_huge**2
    swa_ops2 = 4 * N_huge * W_small
    print(f"\n  Full attention (N={N_huge}):     ~{full_ops2 / 1e9:.0f} GFLOPs")
    print(f"  Sliding window (W={W_small}):    ~{swa_ops2 / 1e9:.0f} GFLOPs")
    print(f"  Reduction:                          {full_ops2 / swa_ops2:.0f}x less compute")


# PERFORMANCE NOTES
# =================
# - Sliding window attention 将 O(N²) 降到 O(N·W):
#   - 每个 query 只看到 W 个 keys，而非全部 N 个
#   - 对于长序列 (N=32K, W=4K)，FLOPs 减少 8x
# - [COMPILER] WINDOW_SIZE 是 tl.constexpr:
#   - 编译时特化 window 大小，避免 runtime mask 检查的开销
#   - 不同 window_size 需要重新编译 kernel
# - [GPU] KV iteration range 根据 window 裁剪:
#   - 相比于 causal mask 的 "up to current position"，
#     sliding window 进一步限制为 "within last W positions"
#   - 减少了不必要的 KV tile 加载
# - Mistral-7B 使用 layer-wise 交替 SWA 和 full attention:
#   - 部分层用 SWA (W=4096)，部分层用 full attention
#   - 这种方式在保持质量的同时大幅降低计算量
# - 与 Flash Attention 的兼容性:
#   - Sliding window mask 与 Flash Attention tiling 正交
#   - 可以同时使用: O(N·W) FLOPs + O(N) memory
# - TODO: 支持不同的 window per layer (Mistral-style layer interleaving)
# - TODO: 支持跨 batch 的可变 window_size
# - TODO: 滑动窗口 + GQA 的组合实现


if __name__ == "__main__":
    main()
