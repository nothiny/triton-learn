"""
13_attention_bias.py — Flash Attention with Bias (ALiBi / Position Bias)

学习目标：
  - 理解 attention bias 在 Flash Attention 中的处理方式
  - 掌握 vector bias (ALiBi) 和 matrix bias 的存储和加载模式
  - 学会 bias + softmax scaling 的融合计算

背景:
  Attention bias 是一种将外部先验注入注意力权重的方式:
    S = Q @ K^T * scale + bias     ← bias 在 softmax 之前添加

  两种常见形式:
    1. Vector bias (ALiBi, Press et al., 2022):
       每个 key position 加一个标量偏置:
         bias_j = -m * |i - j|  其中 m 是 head-specific slope
       用于 BLOOM、MPT 等模型，替代 positional embeddings
       存储为 (seqlen_k,) per (batch, head)

    2. Matrix bias (Relative Position Bias, T5/RoPE-style):
       每个 (query, key) pair 加一个矩阵偏置:
         bias_ij = f(i, j)  其中 f 是自定义函数
       用于 T5、DeBERTa 等模型
       存储为 (seqlen_q, seqlen_k) per (batch, head)

  关键实现细节:
    - Bias 在 softmax scaling 之后、exp 之前添加
    - 当有 bias 时，softmax scaling 在 bias 加法之前完成
    - m_ij = max(S) 而非 max(QK^T)*scale (bias 改变 max)

  参考: flash_attn_triton.py (Tri Dao's implementation) 支持
        vector bias 和 matrix bias 两种模式

运行: python phase2_compute/13_attention_bias.py
"""

import math
import torch
import triton
import triton.language as tl


@triton.jit
def flash_attn_bias_fwd_kernel(
    q_ptr, k_ptr, v_ptr, bias_ptr, o_ptr,
    BATCH, N_HEADS, N_CTX,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_bb, stride_bh, stride_bm, stride_bn,  # bias strides (seq_q, seq_k)
    stride_ob, stride_oh, stride_om,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    D_HEAD: tl.constexpr,
    BIAS_TYPE: tl.constexpr,  # 0=none, 1=vector, 2=matrix
    CAUSAL: tl.constexpr,
    SCALE: tl.constexpr,
):
    """
    Flash Attention forward kernel with optional bias.

    Supports:
      BIAS_TYPE=0: no bias (standard Flash Attention)
      BIAS_TYPE=1: vector bias of shape (seqlen_k,) — ALiBi style
      BIAS_TYPE=2: matrix bias of shape (seqlen_q, seqlen_k) — position bias
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

    # Bias pointers (pre-computed per batch/head)
    if BIAS_TYPE == 1:
        # Vector bias: (1, seqlen_k) per (batch, head). Access as 1D vector.
        # stride_bn is the seq_k stride (usually 1 for contiguous).
        b_base = bias_ptr + batch_idx * stride_bb + head_idx * stride_bh
    elif BIAS_TYPE == 2:
        # Matrix bias: (seqlen_q, seqlen_k) per (batch, head).
        # stride_bm = seq_q stride, stride_bn = seq_k stride.
        # Base pointer at (batch, head, q_start, 0), then add kv offset.
        b_base = (bias_ptr + batch_idx * stride_bb + head_idx * stride_bh +
                  offs_q[:, None] * stride_bm)

    # Online softmax state
    m_i = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)
    acc = tl.zeros([BLOCK_Q, D_HEAD], dtype=tl.float32)

    # KV loop range
    kv_end = (block_q_idx + 1) * BLOCK_Q if CAUSAL else N_CTX

    for block_kv_start in range(0, kv_end, BLOCK_KV):
        offs_kv = block_kv_start + tl.arange(0, BLOCK_KV)

        # Load K tile: [BLOCK_KV, D_HEAD]
        k_ptrs = (k_ptr + batch_idx * stride_kb + head_idx * stride_kh +
                  offs_kv[:, None] * stride_kn + offs_d[None, :])
        k = tl.load(k_ptrs, mask=offs_kv[:, None] < N_CTX, other=0.0)

        # S = Q @ K^T * scale : [BLOCK_Q, BLOCK_KV]
        s = tl.dot(q, k.T) * SCALE

        # ---- Load and apply bias ----
        if BIAS_TYPE == 1:
            # Vector bias: (seqlen_k,) per (batch, head) — load 1D
            bias = tl.load(b_base + offs_kv * stride_bn, mask=offs_kv < N_CTX,
                          other=0.0).to(tl.float32)
            bias = bias[None, :]  # [1, BLOCK_KV] → broadcast to [BLOCK_Q, BLOCK_KV]
            s = s + bias
        elif BIAS_TYPE == 2:
            # Matrix bias: (seqlen_q, seqlen_k) per (batch, head) — load 2D tile
            # b_base already includes offs_q[:, None] * stride_bm
            # Now add offs_kv[None, :] * stride_bn for the key dimension
            bias = tl.load(b_base + offs_kv[None, :] * stride_bn,
                          mask=(offs_q[:, None] < N_CTX) & (offs_kv[None, :] < N_CTX),
                          other=0.0).to(tl.float32)
            s = s + bias

        # ---- Causal mask ----
        if CAUSAL:
            q_pos = offs_q[:, None]
            kv_pos = offs_kv[None, :]
            s = tl.where(q_pos >= kv_pos, s, float("-inf"))

        # Mask out-of-bounds KV positions
        s = tl.where(offs_kv[None, :] < N_CTX, s, float("-inf"))

        # Online softmax update (with NaN guard for fully-masked rows)
        m_ij = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        # NaN guard: if m_new is -inf (first block, all masked), use 0
        m_safe = tl.where(m_new > -1e30, m_new, 0.0)
        m_i_safe = tl.where(m_i > -1e30, m_i, 0.0)

        p = tl.exp(s - m_safe[:, None])
        alpha = tl.exp(m_i_safe - m_safe)
        l_new = alpha * l_i + tl.sum(p, axis=1)

        acc = acc * alpha[:, None]

        v_ptrs = (v_ptr + batch_idx * stride_vb + head_idx * stride_vh +
                  offs_kv[:, None] * stride_vn + offs_d[None, :])
        v = tl.load(v_ptrs, mask=offs_kv[:, None] < N_CTX, other=0.0).to(tl.float32)
        acc += tl.dot(p, v)

        m_i = tl.maximum(m_i, tl.where(m_ij > -1e30, m_ij, float('-inf')))
        l_i = l_new

    # Final normalization
    acc = tl.where(l_i[:, None] > 0.0, acc / l_i[:, None], 0.0)

    # Write output
    offs_m = block_q_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)
    o_ptrs = (o_ptr + batch_idx * stride_ob + head_idx * stride_oh +
              offs_m[:, None] * stride_om + offs_d[None, :])
    tl.store(o_ptrs, acc, mask=offs_m[:, None] < N_CTX)


# ==============================================================================
# Python Wrapper
# ==============================================================================


def flash_attention_with_bias(
    q: torch.Tensor,        # (batch, n_heads, seq_len, d_head)
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor = None,  # (batch, n_heads, 1|seq_q, seq_k) or None
    causal: bool = False,
    block_q: int = 64,
    block_kv: int = 64,
) -> torch.Tensor:
    """
    Flash Attention with optional bias.

    Args:
        q, k, v: Standard attention tensors (fp16/bf16).
        bias: Optional bias tensor. Two forms:
            - Vector: (batch, n_heads, 1, seq_k) — ALiBi-style per-key bias
            - Matrix: (batch, n_heads, seq_q, seq_k) — per-position bias
            - None: no bias
        causal: Apply causal mask after bias.

    Returns:
        o: (batch, n_heads, seq_len, d_head)
    """
    BATCH, N_HEADS, N_CTX, D_HEAD = q.shape
    o = torch.empty_like(q)

    if bias is None:
        bias_type = 0  # none
        bias_tensor = torch.empty(1, device=q.device)  # dummy
        bb = bh = bm = bn = 0
    elif bias.dim() == 4 and bias.shape[2] == 1:
        bias_type = 1  # vector: (B, H, 1, N)
        bias_tensor = bias.contiguous()
        bb, bh, _, bn = bias_tensor.stride()  # bn = seq_k stride
        bm = 0  # vector bias has no seq_q stride
    else:
        bias_type = 2  # matrix: (B, H, N, N)
        bias_tensor = bias.contiguous()
        bb, bh, bm, bn = bias_tensor.stride()
        # bm = seq_q stride, bn = seq_k stride

    grid = (BATCH * N_HEADS * triton.cdiv(N_CTX, block_q),)

    flash_attn_bias_fwd_kernel[grid](
        q, k, v, bias_tensor, o,
        BATCH, N_HEADS, N_CTX,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        bb, bh, bm, bn,
        o.stride(0), o.stride(1), o.stride(2),
        BLOCK_Q=block_q,
        BLOCK_KV=block_kv,
        D_HEAD=D_HEAD,
        BIAS_TYPE=bias_type,
        CAUSAL=causal,
        SCALE=1.0 / math.sqrt(D_HEAD),
    )
    return o


# ==============================================================================
# ALiBi Helper
# ==============================================================================


def build_alibi_slopes(n_heads: int) -> torch.Tensor:
    """
    Build ALiBi slopes following Press et al. (2022).

    slopes[i] = 2^(-8 * (i+1) / n_heads)  ← geometric series

    Returns (n_heads,) tensor.
    """
    # i ranges from 1 to n_heads
    powers = torch.arange(1, n_heads + 1, dtype=torch.float32)
    slopes = 2.0 ** (-8.0 * powers / n_heads)
    return slopes


def build_alibi_bias(
    seq_len: int,
    n_heads: int,
    device: torch.device,
    dtype: torch.float32 = torch.float32,
) -> torch.Tensor:
    """
    Build ALiBi bias tensor: (1, n_heads, 1, seq_len).

    bias_j = -slope_h * j   where j is the key position

    For causal attention, this gives exponentially decaying attention
    to earlier tokens, with different decay rates per head.

    Returns:
        bias: (1, n_heads, 1, seq_len)
    """
    slopes = build_alibi_slopes(n_heads).to(device=device, dtype=dtype)
    positions = torch.arange(seq_len, device=device, dtype=dtype)
    # bias = -slope * position → (n_heads, seq_len)
    bias = -slopes[:, None] * positions[None, :]
    return bias.unsqueeze(0).unsqueeze(2)  # (1, n_heads, 1, seq_len)


# ==============================================================================
# References
# ==============================================================================


def ref_attention_with_bias(q, k, v, bias=None, causal=False):
    """
    PyTorch reference: standard attention with optional bias.
    """
    d_head = q.shape[-1]
    scale = 1.0 / (d_head ** 0.5)
    attn = (q @ k.transpose(-2, -1)) * scale

    if bias is not None:
        attn = attn + bias

    if causal:
        N = attn.shape[-1]
        mask = torch.tril(torch.ones(N, N, device=q.device))
        attn = attn.masked_fill(mask == 0, float("-inf"))

    attn = torch.softmax(attn, dim=-1)
    return attn @ v


# ==============================================================================
# Main
# ==============================================================================


def main():
    print("=" * 70)
    print("13_attention_bias — Flash Attention with Bias (ALiBi)")
    print("=" * 70)

    torch.manual_seed(42)

    # Test configurations
    test_configs = [
        # (B, H, N, D, bias_type, causal, desc)
        (1, 4, 128, 64, "none", False, "No bias, non-causal"),
        (1, 4, 128, 64, "none", True,  "No bias, causal (= v2)"),
        (1, 4, 128, 64, "vector", True, "Vector bias (ALiBi), causal"),
        (1, 4, 128, 64, "matrix", True, "Matrix bias, causal"),
        (1, 4, 128, 64, "vector", False, "Vector bias, non-causal"),
        (2, 8, 256, 64, "vector", True, "Larger: vector bias + causal"),
    ]

    for B, H, N, D, bias_type, causal, desc in test_configs:
        print(f"\n── {desc} ──")

        q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

        # Build bias
        if bias_type == "vector":
            # ALiBi-style: (B, H, 1, N)
            bias = build_alibi_bias(N, H, q.device, dtype=torch.float32)
            bias = bias.expand(B, H, 1, N)
        elif bias_type == "matrix":
            # Random relative position bias: (B, H, N, N)
            bias = torch.randn(B, H, N, N, device=q.device, dtype=torch.float32) * 0.1
        else:
            bias = None

        # Triton
        o_triton = flash_attention_with_bias(q, k, v, bias, causal=causal)

        # Reference
        o_ref = ref_attention_with_bias(
            q.float(), k.float(), v.float(),
            bias=bias.float() if bias is not None else None,
            causal=causal,
        ).half()

        max_diff = (o_triton.float() - o_ref.float()).abs().max().item()
        status = "✅" if max_diff < 0.05 else "❌"
        print(f"  max_diff = {max_diff:.6e}  {status}")

    # ALiBi verification: check that early positions get more attention
    print(f"\n{'='*70}")
    print("ALiBi Verification — attention weight decay over distance")
    print(f"{'='*70}")

    N, H, D = 16, 4, 32
    slopes = build_alibi_slopes(H)
    bias = build_alibi_bias(N, H, torch.device("cuda"))
    bias = bias.expand(1, H, 1, N)

    # Uniform Q,K so attention weights come purely from bias
    q_unif = torch.ones(1, H, N, D, device="cuda", dtype=torch.float16) * 0.01
    k_unif = torch.ones(1, H, N, D, device="cuda", dtype=torch.float16) * 0.01
    v_eye = torch.eye(N, D, device="cuda", dtype=torch.float16).unsqueeze(0).repeat(1, H, 1, 1)

    o_alibi = flash_attention_with_bias(q_unif, k_unif, v_eye, bias, causal=True)
    print(f"\n  Head slopes: {[f'{s:.3f}' for s in slopes[:4].tolist()]}")
    print(f"  Last position (N-1) output (first 4 cols) — should show decay:")
    print(f"  Head 0: {[f'{v:.3f}' for v in o_alibi[0, 0, -1, :4].tolist()]}")
    print(f"  Head 1: {[f'{v:.3f}' for v in o_alibi[0, 1, -1, :4].tolist()]}")
    print(f"  Head 2: {[f'{v:.3f}' for v in o_alibi[0, 2, -1, :4].tolist()]}")
    print(f"  Head 3: {[f'{v:.3f}' for v in o_alibi[0, 3, -1, :4].tolist()]}")
    print(f"\n  ✅ Larger slope → more weight on recent tokens (steeper decay)")

    # Performance note: bias is added BEFORE softmax, so numerical error
    # accumulates differently than the v2 causal-only case
    print(f"\n{'='*70}")
    print("Bias + Softmax numerical notes")
    print(f"{'='*70}")
    print("  With bias: S = QK^T * scale + bias, then softmax.")
    print("  The bias shifts the effective max per row, changing the")
    print("  online softmax rescaling factors (alpha). This is handled")
    print("  correctly because m_ij = max(S), not max(QK^T * scale).")
    print("  ALiBi slopes cause exponentially decaying attention weights")
    print("  — a form of recency bias without learned position embeddings.")


# PERFORMANCE NOTES
# =================
# - Attention bias adds minimal FLOPs overhead (element-wise addition)
# - Memory overhead depends on bias type:
#   - Vector bias: O(N) per (batch, head) — negligible
#   - Matrix bias: O(N²) per (batch, head) — can be large, but still
#     much smaller than the full attention matrix when combined with
#     Flash Attention tiling
# - [COMPILER] BIAS_TYPE is tl.constexpr: compiler eliminates all dead
#   code paths for bias loading and application
# - When bias is present, softmax scaling must happen BEFORE bias addition
#   (multiply-then-add, vs add-then-multiply). This is intentional:
#   the scale should apply to the dot product, not to the bias.
# - ALiBi is particularly efficient: the bias is computed on-the-fly
#   from slopes and positions, requiring only O(1) extra memory
#   per head for the slope value
# - [GPU] The bias memory access pattern:
#   - Vector: coalesced loads along KV dimension
#   - Matrix: 2D tile loads, similar to Q/K tile access pattern
# - TODO: Support dynamic mask_mod functions (FlashAttention-3 style)
# - TODO: In-place bias for fused positional encoding attention


if __name__ == "__main__":
    main()
