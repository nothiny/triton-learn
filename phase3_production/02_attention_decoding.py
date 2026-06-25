"""
phase3_production/02_attention_decoding.py - Attention Decoding (Inference)

学习目标:
  - tl.make_block_ptr 处理变长序列 K/V 循环
  - GQA + online softmax 的生产级 pattern
  - 对比 phase2_compute/12_flash_attention_v1.py 的手工指针拼接

运行: python phase3_production/02_attention_decoding.py
"""

import torch
import triton
import triton.language as tl
from triton.testing import do_bench


@triton.autotune(
    configs=[
        triton.Config({"BS": bs, "BV": bv}, num_warps=w, num_stages=s)
        for bs in [32, 64] for bv in [32, 64, 128] for w in [2, 4] for s in [2, 3]
    ],
    key=["D_HEAD", "D_VALUE"],
)
@triton.jit
def attn_decoding_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    B, T, HQ, H,
    D_HEAD, D_VALUE,
    stride_q_b, stride_q_h, stride_q_d,
    stride_k_b, stride_k_t, stride_k_h, stride_k_d,
    stride_v_b, stride_v_t, stride_v_h, stride_v_d,
    stride_o_b, stride_o_h, stride_o_d,
    scale,
    HEAD_RATIO: tl.constexpr,
    D_HEAD_CONST: tl.constexpr,  # D_HEAD as constexpr for tl.arange
    BS: tl.constexpr, BV: tl.constexpr,
):
    """Q[1, B, HQ, D] x K/V[1, B, T, H, D] -> O[1, B, HQ, D]"""
    pid = tl.program_id(0)
    batch_idx = pid // HQ
    hq_idx = pid % HQ
    h_idx = hq_idx // HEAD_RATIO

    # Load full Q (single token, D_HEAD <= 128 fits in registers)
    offs_d = tl.arange(0, D_HEAD_CONST)
    q_ptrs = q_ptr + batch_idx * stride_q_b + hq_idx * stride_q_h + offs_d
    b_q = tl.load(q_ptrs, mask=offs_d < D_HEAD, other=0.0)  # [D_HEAD_CONST]
    b_q = (b_q * scale).to(b_q.dtype)

    # Online softmax state
    m_i = tl.full([1], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([BV], dtype=tl.float32)

    # K/V via block_ptr
    p_k = tl.make_block_ptr(
        base=k_ptr + batch_idx * stride_k_b + h_idx * stride_k_h,
        shape=(T, D_HEAD), strides=(stride_k_t, stride_k_d),
        offsets=(0, 0), block_shape=(BS, D_HEAD_CONST), order=(1, 0))
    p_v = tl.make_block_ptr(
        base=v_ptr + batch_idx * stride_v_b + h_idx * stride_v_h,
        shape=(T, D_VALUE), strides=(stride_v_t, stride_v_d),
        offsets=(0, 0), block_shape=(BS, BV), order=(1, 0))

    # Scan over K/V sequence
    for t_start in range(0, T, BS):
        b_k = tl.load(p_k, boundary_check=(0, 1))  # [BS, BK]
        b_v = tl.load(p_v, boundary_check=(0, 1))  # [BS, BV]

        # QK^T: [D_HEAD] x [BS, D_HEAD] -> sum -> [BS]
        b_s = tl.sum(b_q[None, :].to(tl.float32) * b_k.to(tl.float32), axis=1)
        valid = (t_start + tl.arange(0, BS)) < T
        b_s = tl.where(valid, b_s, float("-inf"))

        # Online softmax
        tile_max = tl.max(b_s, axis=0)
        m_new = tl.maximum(m_i, tile_max)
        rescale = tl.exp(m_i - m_new)
        l_i = l_i * rescale + tl.sum(tl.exp(b_s - m_new), axis=0)
        acc = acc * rescale
        p = tl.exp(b_s - m_new)                    # [BS]
        acc += tl.sum(p[:, None] * b_v.to(tl.float32), axis=0)  # [BV]
        m_i = m_new

        p_k = tl.advance(p_k, (BS, 0))
        p_v = tl.advance(p_v, (BS, 0))

    # Normalize & store
    acc = acc / l_i
    offs_v = tl.arange(0, BV)
    o_ptrs = o_ptr + batch_idx * stride_o_b + hq_idx * stride_o_h + offs_v
    tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), mask=offs_v < D_VALUE)


def attention_decoding(q, k, v, scale=None):
    B, HQ, D_HEAD = q.shape
    _, T, H, _ = k.shape
    D_VALUE = v.shape[-1]
    if scale is None:
        scale = D_HEAD ** -0.5
    o = torch.empty(B, HQ, D_VALUE, device=q.device, dtype=q.dtype)
    attn_decoding_kernel[(B * HQ,)](
        q, k, v, o, B, T, HQ, H, D_HEAD, D_VALUE,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2),
        scale, HEAD_RATIO=HQ // H, D_HEAD_CONST=D_HEAD)
    return o


def ref_attention_decoding(q, k, v, scale=None):
    if scale is None:
        scale = q.shape[-1] ** -0.5
    B, HQ, D = q.shape
    _, T, H, V = v.shape
    G = HQ // H
    out = torch.zeros(B, HQ, V, device=q.device, dtype=torch.float32)
    for b in range(B):
        for hq in range(HQ):
            h = hq // G
            scores = (q[b, hq].float() @ k[b, :, h].float().T) * scale
            attn = torch.softmax(scores, dim=-1)
            out[b, hq] = attn @ v[b, :, h].float()
    return out.to(q.dtype)


def main():
    print("=" * 60)
    print("02_attention_decoding - Decoding Attention (Inference)")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    configs = [
        (1, 8, 2, 64, 64, 512, "GQA 8Q/2KV D=64 T=512"),
        (1, 16, 4, 128, 128, 256, "GQA 16Q/4KV D=128 T=256"),
        (2, 4, 1, 64, 64, 1024, "MQA 4Q/1KV D=64 T=1K"),
        (1, 32, 8, 64, 64, 2048, "GQA 32Q/8KV D=64 T=2K"),
    ]

    for B, HQ, H, D_HEAD, D_V, T, desc in configs:
        q = torch.randn(B, HQ, D_HEAD, device="cuda", dtype=torch.float16)
        k = torch.randn(B, T, H, D_HEAD, device="cuda", dtype=torch.float16)
        v = torch.randn(B, T, H, D_V, device="cuda", dtype=torch.float16)
        scale = D_HEAD ** -0.5

        o_triton = attention_decoding(q, k, v, scale)
        o_ref = ref_attention_decoding(q, k, v, scale)
        max_diff = (o_triton.float() - o_ref.float()).abs().max().item()

        ms_triton = do_bench(lambda: attention_decoding(q, k, v, scale))
        ms_ref = do_bench(lambda: ref_attention_decoding(q, k, v, scale))

        ok = max_diff < 0.05
        print(f"  {desc}: Triton={ms_triton:.4f}ms ref={ms_ref:.4f}ms "
              f"speedup={ms_ref/ms_triton:.1f}x diff={max_diff:.2e} "
              f"{'OK' if ok else 'FAIL'}")


# PERFORMANCE NOTES
# =================
# Decoding attention: Q=1 token, all KV visible
# K/V loaded via tl.make_block_ptr with:
#   - boundary_check for automatic edge masking
#   - tl.advance for loop index progression
#   - compiler can infer access pattern -> better cp.async prefetch
#
# Q loaded via plain pointer (single token, no tiling needed)
# Output stored via plain pointer (1D, simple)


if __name__ == "__main__":
    main()
