"""
17_fused_linear_residual_norm.py — Fused Linear + Residual + Per-Tile LayerNorm

学习目标:
  - 理解 transformer block 中最常见的操作融合模式
  - 掌握 epilogue fusion: 在 accumulator 写回前插入额外计算
  - 学习 per-tile normalization 的简化实现

注意:
  本实现做的是 per-tile LayerNorm（每个 BLOCK_N tile 独立归一化），
  不是标准 LayerNorm（整行归一化）。原因:
  - 每行所有列的数据分布在多次 N-tile 循环中，不全部同时驻留在寄存器
  - 真正跨 tile LayerNorm 需要 2-pass: 先算全局 mean/std，再归一化
  - 这个简化的 per-tile norm 正确演示了 linear+norm fusion 机制

  参考: liger-kernel 的 FusedLinearLayerNorm 使用了完整的跨 block reduction

运行: python phase2_compute/17_fused_linear_residual_norm.py
"""

import torch
import triton
import triton.language as tl
from triton.testing import do_bench


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def fused_linear_norm_kernel(
    x_ptr, w_ptr, bias_ptr,
    residual_ptr,    # (M, N)
    gamma_ptr,       # (N,)
    beta_ptr,        # (N,)
    out_ptr,         # (M, N)
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_rm, stride_rn,
    stride_om, stride_on,
    eps: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    out = PerTileLayerNorm((x @ W + bias + residual), gamma, beta)

    每个 program 处理 M 维的一行，对 N 维的所有 tile 分别 norm。
    每个 N tile 的结果独立，跨 tile 无依赖。
    """
    pid_m = tl.program_id(axis=0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M

    # 有效行数（最后一行可能不满 BLOCK_M）
    valid_rows = tl.minimum(BLOCK_M, M - pid_m * BLOCK_M)

    num_pid_n = tl.cdiv(N, BLOCK_N)

    for pid_n in range(num_pid_n):
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_valid = tl.minimum(BLOCK_N, N - pid_n * BLOCK_N)

        # Linear: acc = x @ W
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        for k in range(0, K, BLOCK_K):
            offs_k = k + tl.arange(0, BLOCK_K)

            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
            x_mask = m_mask[:, None] & (offs_k[None, :] < K)
            x = tl.load(x_ptrs, mask=x_mask, other=0.0)

            w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
            w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
            w = tl.load(w_ptrs, mask=w_mask, other=0.0)

            acc += tl.dot(x, w)

        # + bias (broadcast)
        b_ptrs = bias_ptr + offs_n
        b_mask = offs_n < N
        bias = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += bias[None, :]

        # + residual
        r_ptrs = residual_ptr + offs_m[:, None] * stride_rm + offs_n[None, :] * stride_rn
        r_mask = m_mask[:, None] & (offs_n[None, :] < N)
        residual = tl.load(r_ptrs, mask=r_mask, other=0.0)
        h = acc + residual  # [BLOCK_M, BLOCK_N]

        # ---- Per-tile LayerNorm ----
        h_f32 = h.to(tl.float32)

        # Row-wise mean 和 rstd（只对本 tile 内的列）
        h_sum = tl.sum(h_f32, axis=1)  # [BLOCK_M]
        h_mean = h_sum / n_valid

        h_centered = h_f32 - h_mean[:, None]
        h_sq_sum = tl.sum(h_centered * h_centered, axis=1)
        h_var = h_sq_sum / n_valid
        h_rstd = 1.0 / tl.sqrt(h_var + eps)

        # Normalize + affine
        h_norm = h_centered * h_rstd[:, None]

        gamma = tl.load(gamma_ptr + offs_n, mask=offs_n < N, other=1.0)
        beta = tl.load(beta_ptr + offs_n, mask=offs_n < N, other=0.0)
        out = h_norm * gamma[None, :] + beta[None, :]

        # Store
        out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        out_mask = m_mask[:, None] & (offs_n[None, :] < N)
        tl.store(out_ptrs, out, mask=out_mask)


def fused_linear_residual_norm(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    M, K = x.shape
    K2, N = w.shape
    assert K == K2
    assert residual.shape == (M, N)
    assert gamma.shape == (N,) and beta.shape == (N,)

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)

    fused_linear_norm_kernel[grid](
        x, w, bias, residual, gamma, beta, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        residual.stride(0), residual.stride(1),
        out.stride(0), out.stride(1),
        eps=eps,
    )
    return out


def ref_per_tile_linear_residual_norm(
    x, w, bias, residual, gamma, beta, BLOCK_N=128, eps=1e-5
) -> torch.Tensor:
    """Per-tile LayerNorm reference (matching the Triton kernel behavior)"""
    h = x @ w + bias + residual
    M, N = h.shape
    out = torch.empty_like(h)

    for n_start in range(0, N, BLOCK_N):
        n_end = min(n_start + BLOCK_N, N)
        h_tile = h[:, n_start:n_end].float()
        g_tile = gamma[n_start:n_end].float()
        b_tile = beta[n_start:n_end].float()

        mean = h_tile.mean(dim=-1, keepdim=True)
        var = h_tile.var(dim=-1, keepdim=True, unbiased=False)
        rstd = 1.0 / torch.sqrt(var + eps)
        normed = (h_tile - mean) * rstd
        out[:, n_start:n_end] = (normed * g_tile + b_tile).to(h.dtype)

    return out


def main():
    print("=" * 60)
    print("17_fused_linear_residual_norm — Fused Linear + Residual + Per-Tile Norm")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    configs = [
        (128, 256, 128),
        (512, 512, 256),
        (1024, 1024, 512),
        (2048, 2048, 1024),
    ]

    for M, K, N in configs:
        x = torch.randn(M, K, device="cuda", dtype=torch.float16)
        w = torch.randn(K, N, device="cuda", dtype=torch.float16)
        bias = torch.randn(N, device="cuda", dtype=torch.float16)
        residual = torch.randn(M, N, device="cuda", dtype=torch.float16)
        gamma = torch.randn(N, device="cuda", dtype=torch.float32)
        beta = torch.randn(N, device="cuda", dtype=torch.float32)

        # Correctness: compare against per-tile norm (same behavior as kernel)
        out_fused = fused_linear_residual_norm(x, w, bias, residual, gamma, beta)
        out_ref = ref_per_tile_linear_residual_norm(x, w, bias, residual, gamma, beta)
        max_diff = (out_fused.float() - out_ref.float()).abs().max().item()

        # Performance: Triton fused vs cuBLAS (matmul only) + separate norm
        ms_fused = do_bench(lambda: fused_linear_residual_norm(
            x, w, bias, residual, gamma, beta))
        ms_cublas_mm = do_bench(lambda: x @ w)  # cuBLAS matmul only (不含 residual+norm)

        flops = 2 * M * N * K
        tflops = flops / (ms_fused * 1e-3) / 1e12

        status = "✅" if max_diff < 0.05 else "❌"
        print(f"  {M}×{K}×{N}: fused={ms_fused:.4f}ms  "
              f"cuBLAS(mm)={ms_cublas_mm:.4f}ms  "
              f"diff={max_diff:.2e}  {status}")


# PERFORMANCE NOTES
# =================
# - 本 fusion 的核心价值: 避免 Linear output 写 HBM
# - 但这里的 LayerNorm 是 per-tile 的（简化版），不是标准 LayerNorm
# - 真正的 fused linear+norm 需要 2-pass:
#   - Pass 1: compute full-row mean/std (跨所有 N tiles)
#   - Pass 2: normalize each tile
#   - 可以用 online Welford 算法在 1-pass 中完成
# - 生产实现参考: liger-kernel LigerFusedLinearLayerNorm


if __name__ == "__main__":
    main()
