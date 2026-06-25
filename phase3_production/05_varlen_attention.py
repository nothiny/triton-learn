"""
phase3_production/05_varlen_attention.py — Variable-Length Attention (cu_seqlens)

学习目标:
  - cu_seqlens 变长序列处理
  - block_ptr 配合动态序列范围
  - 生产级 varlen attention 的 core pattern

cu_seqlens: [0, len0, len0+len1, ...] — 每个 batch 的累积序列偏移
  bos = cu_seqlens[b]     — batch b 的起始位置
  eos = cu_seqlens[b+1]   — batch b 的结束位置
  T = eos - bos           — batch b 的实际序列长度

运行: python phase3_production/05_varlen_attention.py
"""

import torch
import triton
import triton.language as tl
from triton.testing import do_bench


@triton.autotune(
    configs=[
        triton.Config({"BQ": bq, "BK": bk, "BV": bv}, num_warps=w, num_stages=s)
        for bq in [32, 64] for bk in [64, 128]
        for bv in [64, 128] for w in [4, 8] for s in [2, 3]
    ],
    key=["D_HEAD"],
)
@triton.jit
def varlen_attn_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    cu_seqlens_ptr,
    H, HQ, max_seqlen, D_HEAD,
    # Full strides: (B, H, N, D)
    stride_q_b, stride_q_h, stride_q_m, stride_q_d,
    stride_k_b, stride_k_h, stride_k_n, stride_k_d,
    stride_v_b, stride_v_h, stride_v_n, stride_v_d,
    stride_o_b, stride_o_h, stride_o_m, stride_o_d,
    scale,
    G: tl.constexpr,
    BQ: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
):
    """Varlen FlashAttention with cu_seqlens."""
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_v = tl.program_id(2)

    batch_idx = pid_bh // HQ
    hq_idx = pid_bh % HQ
    h_idx = hq_idx // G

    # ── Read sequence bounds ─────────────────────────────────
    bos = tl.load(cu_seqlens_ptr + batch_idx).to(tl.int32)
    eos = tl.load(cu_seqlens_ptr + batch_idx + 1).to(tl.int32)
    T = eos - bos

    q_start = pid_q * BQ
    if q_start >= T:
        return

    # ── Q tile: [BQ, BK] ──────────────────────────────────────
    p_q = tl.make_block_ptr(
        base=q_ptr + batch_idx * stride_q_b + hq_idx * stride_q_h,
        shape=(max_seqlen, D_HEAD),
        strides=(stride_q_m, stride_q_d),
        offsets=(q_start, 0),
        block_shape=(BQ, BK), order=(1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)

    # ── O tile ────────────────────────────────────────────────
    p_o = tl.make_block_ptr(
        base=o_ptr + batch_idx * stride_o_b + hq_idx * stride_o_h,
        shape=(max_seqlen, D_HEAD),
        strides=(stride_o_m, stride_o_d),
        offsets=(q_start, pid_v * BV),
        block_shape=(BQ, BV), order=(1, 0))

    # ── Online softmax ───────────────────────────────────────
    m_i = tl.full([BQ, 1], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BQ, 1], dtype=tl.float32)
    acc = tl.zeros([BQ, BV], dtype=tl.float32)

    num_kv_blocks = tl.cdiv(T, BK)
    for kv_block in range(num_kv_blocks):
        k_start = kv_block * BK

        p_k = tl.make_block_ptr(
            base=k_ptr + batch_idx * stride_k_b + h_idx * stride_k_h,
            shape=(max_seqlen, D_HEAD),
            strides=(stride_k_n, stride_k_d),
            offsets=(k_start, 0),
            block_shape=(BK, BK), order=(1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))

        p_v = tl.make_block_ptr(
            base=v_ptr + batch_idx * stride_v_b + h_idx * stride_v_h,
            shape=(max_seqlen, D_HEAD),
            strides=(stride_v_n, stride_v_d),
            offsets=(k_start, pid_v * BV),
            block_shape=(BK, BV), order=(1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))

        # QK^T
        b_s = tl.dot(b_q.to(tl.float16), b_k.to(tl.float16).T).to(tl.float32)

        # Mask: K positions within T (Q boundary handled by boundary_check)
        k_pos = k_start + tl.arange(0, BK)
        b_s = tl.where(k_pos[None, :] < T, b_s, float("-inf"))

        # Online softmax
        m_new = tl.maximum(m_i, tl.max(b_s, axis=1, keep_dims=True))
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(tl.exp(b_s - m_new), axis=1, keep_dims=True)
        acc = acc * alpha
        p = tl.exp(b_s - m_new)
        acc += tl.dot(p.to(tl.float16), b_v.to(tl.float16)).to(tl.float32)
        m_i = m_new

    acc = acc / l_i
    tl.store(p_o, acc.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))


def varlen_attention(q, k, v, cu_seqlens, scale=None):
    B, HQ, max_seqlen, D_HEAD = q.shape
    _, H, _, _ = k.shape
    G = HQ // H
    if scale is None:
        scale = D_HEAD ** -0.5
    o = torch.empty_like(q)

    grid = lambda meta: (
        triton.cdiv(max_seqlen, meta["BQ"]),
        B * HQ,
        triton.cdiv(D_HEAD, meta["BV"]))
    varlen_attn_kernel[grid](
        q, k, v, o, cu_seqlens, H, HQ, max_seqlen, D_HEAD,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        scale, G=G)
    return o


def ref_varlen_attention(q, k, v, cu_seqlens, scale=None):
    if scale is None:
        scale = q.shape[-1] ** -0.5
    B, HQ, _, D = q.shape
    _, H, _, _ = k.shape
    G = HQ // H
    out = torch.zeros_like(q, dtype=torch.float32)
    for b in range(B):
        T = int(cu_seqlens[b + 1] - cu_seqlens[b])
        for hq in range(HQ):
            h = hq // G
            scores = (q[b, hq, :T].float() @ k[b, h, :T].float().T) * scale
            attn = torch.softmax(scores, dim=-1)
            out[b, hq, :T] = attn @ v[b, h, :T].float()
    return out.to(q.dtype)


def main():
    print("=" * 60)
    print("05_varlen_attention — Variable-Length Attention")
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    configs = [
        (1, 4, 1, 128, 64, [128], "T=128 (full)"),
        (2, 8, 2, 128, 64, [64, 128], "2 batch T=[64,128]"),
        (1, 4, 1, 128, 64, [128], "GQA T=128"),
    ]

    for B, HQ, H, N, D, seqlens, desc in configs:
        cu_seqlens = torch.tensor([0] + list(torch.tensor(seqlens).cumsum(0)),
                                  device="cuda", dtype=torch.int32)
        q = torch.randn(B, HQ, N, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

        o_triton = varlen_attention(q, k, v, cu_seqlens)
        o_ref = ref_varlen_attention(q, k, v, cu_seqlens)
        max_diff = (o_triton.float() - o_ref.float()).abs().max().item()

        ms_triton = do_bench(lambda: varlen_attention(q, k, v, cu_seqlens))
        ms_ref = do_bench(lambda: ref_varlen_attention(q, k, v, cu_seqlens))

        ok = max_diff < 2.0
        print(f"  {desc}: Triton={ms_triton:.4f}ms ref={ms_ref:.4f}ms "
              f"speedup={ms_ref/ms_triton:.1f}x diff={max_diff:.2e} "
              f"{'OK' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
