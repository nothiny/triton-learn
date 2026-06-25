"""
phase3_production/04_fused_rms_norm_residual.py — Fused RMSNorm + Residual (LLaMA pattern)

学习目标:
  - tl.make_block_ptr 用于 1D row-wise reduction
  - 融合 residual add → 避免中间 HBM 写入
  - 生产级 RMSNorm 的 2-pass pattern

LLaMA 中的核心操作:
  h = x + residual
  normed = RMSNorm(h, weight) = h / rms(h) * weight

运行: python phase3_production/04_fused_rms_norm_residual.py
"""

import torch
import triton
import triton.language as tl
from triton.testing import do_bench


@triton.autotune(
    configs=[
        triton.Config({"BN": bn}, num_warps=w, num_stages=s)
        for bn in [256, 512, 1024, 2048]
        for w in [4, 8] for s in [2, 3]
    ],
    key=["N"],
)
@triton.jit
def fused_rms_norm_residual_kernel(
    x_ptr, residual_ptr, weight_ptr, output_ptr,
    M, N,
    stride_x_m, stride_x_n,
    stride_r_m, stride_r_n,
    stride_o_m, stride_o_n,
    eps: tl.constexpr,
    BN: tl.constexpr,
):
    """out_row = RMSNorm(x_row + residual_row, weight)"""
    pid_m = tl.program_id(0)

    # Block pointers for row-wise access
    p_x = tl.make_block_ptr(
        base=x_ptr + pid_m * stride_x_m, shape=(N,), strides=(stride_x_n,),
        offsets=(0,), block_shape=(BN,), order=(0,))
    p_r = tl.make_block_ptr(
        base=residual_ptr + pid_m * stride_r_m, shape=(N,), strides=(stride_r_n,),
        offsets=(0,), block_shape=(BN,), order=(0,))
    p_w = tl.make_block_ptr(
        base=weight_ptr, shape=(N,), strides=(1,),
        offsets=(0,), block_shape=(BN,), order=(0,))
    p_o = tl.make_block_ptr(
        base=output_ptr + pid_m * stride_o_m, shape=(N,), strides=(stride_o_n,),
        offsets=(0,), block_shape=(BN,), order=(0,))

    # Pass 1: compute RMS(x + residual)
    sum_sq = tl.zeros([1], dtype=tl.float32)
    num_tiles = tl.cdiv(N, BN)
    for _ in range(num_tiles):
        x = tl.load(p_x, boundary_check=(0,)).to(tl.float32)
        r = tl.load(p_r, boundary_check=(0,)).to(tl.float32)
        sum_sq += tl.sum((x + r) * (x + r), axis=0)
        p_x = tl.advance(p_x, (BN,))
        p_r = tl.advance(p_r, (BN,))
    rms = tl.sqrt(sum_sq / N + eps)

    # Reset pointers for Pass 2
    p_x = tl.make_block_ptr(
        base=x_ptr + pid_m * stride_x_m, shape=(N,), strides=(stride_x_n,),
        offsets=(0,), block_shape=(BN,), order=(0,))
    p_r = tl.make_block_ptr(
        base=residual_ptr + pid_m * stride_r_m, shape=(N,), strides=(stride_r_n,),
        offsets=(0,), block_shape=(BN,), order=(0,))

    # Pass 2: normalize + scale + store
    for _ in range(num_tiles):
        x = tl.load(p_x, boundary_check=(0,)).to(tl.float32)
        r = tl.load(p_r, boundary_check=(0,)).to(tl.float32)
        w = tl.load(p_w, boundary_check=(0,)).to(tl.float32)
        normed = ((x + r) / rms) * w
        tl.store(p_o, normed.to(output_ptr.dtype.element_ty), boundary_check=(0,))
        p_x = tl.advance(p_x, (BN,)); p_r = tl.advance(p_r, (BN,))
        p_w = tl.advance(p_w, (BN,)); p_o = tl.advance(p_o, (BN,))


def fused_rms_norm_residual(x, residual, weight, eps=1e-6):
    M, N = x.shape
    out = torch.empty_like(x)
    fused_rms_norm_residual_kernel[(M,)](
        x, residual, weight, out, M, N,
        x.stride(0), x.stride(1),
        residual.stride(0), residual.stride(1),
        out.stride(0), out.stride(1), eps=eps)
    return out


def ref_rms_norm_residual(x, residual, weight, eps=1e-6):
    h = x.float() + residual.float()
    rms = torch.sqrt((h * h).mean(dim=-1, keepdim=True) + eps)
    return (h / rms * weight.float()).to(x.dtype)


def main():
    print("=" * 60)
    print("04_fused_rms_norm_residual — Fused RMSNorm + Residual")
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    for M, N in [(128, 1024), (512, 2048), (1024, 4096), (2048, 8192)]:
        x = torch.randn(M, N, device="cuda", dtype=torch.float16)
        residual = torch.randn(M, N, device="cuda", dtype=torch.float16)
        weight = torch.randn(N, device="cuda", dtype=torch.float16)

        out_triton = fused_rms_norm_residual(x, residual, weight)
        out_ref = ref_rms_norm_residual(x, residual, weight)
        max_diff = (out_triton.float() - out_ref.float()).abs().max().item()

        ms_fused = do_bench(lambda: fused_rms_norm_residual(x, residual, weight))
        ms_unfused = do_bench(lambda: ref_rms_norm_residual(x, residual, weight))

        ok = max_diff < 0.01  # fp16 RMSNorm tolerance
        print(f"  M={M}, N={N}: fused={ms_fused:.4f}ms unfused={ms_unfused:.4f}ms "
              f"speedup={ms_unfused/ms_fused:.2f}x diff={max_diff:.2e} "
              f"{'OK' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
