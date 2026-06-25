"""
16_weight_gradient.py — MatMul 的权重梯度 dW = X^T @ dY

学习目标:
  - 理解 matmul backward 的计算模式
  - 掌握规约维度变化时的 tiling 调整
  - 学会如何处理隐式转置在 tl.dot 中的用法

计算:
  前向:  Y = X @ W    (M×K) @ (K×N) → (M×N)
  反向:  dW = X^T @ dY   (K×M) @ (M×N) → (K×N)
         dX = dY @ W^T   (M×N) @ (N×K) → (M×K)

  dW 的规约维度是 M（不再是 K），dX 的规约维度是 N（不再是 K）。

Grid 设计:
  dW: (cdiv(K, BK), cdiv(N, BN)) — 跟 forward matmul 一样是 2D grid
  dX: (cdiv(M, BM), cdiv(K, BK)) — 跟 forward matmul 一样是 2D grid

运行: python phase2_compute/16_weight_gradient.py
"""

import torch
import triton
import triton.language as tl
from triton.testing import do_bench


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_K": 64, "BLOCK_N": 64, "BLOCK_M": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_K": 64, "BLOCK_N": 128, "BLOCK_M": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_K": 128, "BLOCK_N": 128, "BLOCK_M": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_K": 128, "BLOCK_N": 128, "BLOCK_M": 64}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def weight_gradient_kernel(
    x_ptr,      # (M, K)
    dy_ptr,     # (M, N)
    dw_ptr,     # (K, N)
    M, N, K,
    stride_xm, stride_xk,
    stride_dym, stride_dyn,
    stride_dwk, stride_dwn,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """
    dW = X^T @ dY

    dW[k, n] = sum_m X[m, k] * dY[m, n]
    """
    pid_k = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_K, BLOCK_N], dtype=tl.float32)

    for m in range(0, M, BLOCK_M):
        offs_m = m + tl.arange(0, BLOCK_M)

        # Load X^T tile: [BLOCK_K, BLOCK_M]
        # X^T[k, m] = X[m, k]
        # 通过交换 offs 的维度来实现转置: 第一维是 k (BLOCK_K)，第二维是 m (BLOCK_M)
        x_ptrs = x_ptr + offs_m[None, :] * stride_xm + offs_k[:, None] * stride_xk
        x_mask = (offs_m[None, :] < M) & (offs_k[:, None] < K)
        x_t = tl.load(x_ptrs, mask=x_mask, other=0.0)  # [BLOCK_K, BLOCK_M]

        # Load dY tile: [BLOCK_M, BLOCK_N]
        dy_ptrs = dy_ptr + offs_m[:, None] * stride_dym + offs_n[None, :] * stride_dyn
        dy_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        dy_tile = tl.load(dy_ptrs, mask=dy_mask, other=0.0)  # [BLOCK_M, BLOCK_N]

        # acc += X^T @ dY: [BK, BM] @ [BM, BN] → [BK, BN]
        acc += tl.dot(x_t, dy_tile)

    dw_ptrs = dw_ptr + offs_k[:, None] * stride_dwk + offs_n[None, :] * stride_dwn
    dw_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
    tl.store(dw_ptrs, acc, mask=dw_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 64, "BLOCK_N": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 128, "BLOCK_N": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 128, "BLOCK_N": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def input_gradient_kernel(
    dy_ptr, w_ptr, dx_ptr,
    M, N, K,
    stride_dym, stride_dyn,
    stride_wk, stride_wn,
    stride_dxm, stride_dxk,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    dX = dY @ W^T

    dX[m, k] = sum_n dY[m, n] * W[k, n]
    """
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)

    for n in range(0, N, BLOCK_N):
        offs_n = n + tl.arange(0, BLOCK_N)

        # Load dY tile: [BLOCK_M, BLOCK_N]
        dy_ptrs = dy_ptr + offs_m[:, None] * stride_dym + offs_n[None, :] * stride_dyn
        dy_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        dy_tile = tl.load(dy_ptrs, mask=dy_mask, other=0.0)  # [BM, BN]

        # Load W^T tile: [BLOCK_N, BLOCK_K]
        # W^T[n, k] = W[k, n]
        w_ptrs = w_ptr + offs_k[None, :] * stride_wk + offs_n[:, None] * stride_wn
        w_mask = (offs_k[None, :] < K) & (offs_n[:, None] < N)
        w_t = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BN, BK]

        # dX += dY @ W^T: [BM, BN] @ [BN, BK] → [BM, BK]
        acc += tl.dot(dy_tile, w_t)

    dx_ptrs = dx_ptr + offs_m[:, None] * stride_dxm + offs_k[None, :] * stride_dxk
    dx_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    tl.store(dx_ptrs, acc, mask=dx_mask)


def weight_gradient(x: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    """dW = X^T @ dY"""
    M, K = x.shape
    M2, N = dy.shape
    assert M == M2

    dw = torch.empty((K, N), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(K, meta["BLOCK_K"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )
    weight_gradient_kernel[grid](
        x, dy, dw,
        M, N, K,
        x.stride(0), x.stride(1),
        dy.stride(0), dy.stride(1),
        dw.stride(0), dw.stride(1),
    )
    return dw


def input_gradient(dy: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """dX = dY @ W^T"""
    M, N = dy.shape
    K, N2 = w.shape
    assert N == N2

    dx = torch.empty((M, K), device=dy.device, dtype=dy.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(K, meta["BLOCK_K"]),
    )
    input_gradient_kernel[grid](
        dy, w, dx,
        M, N, K,
        dy.stride(0), dy.stride(1),
        w.stride(0), w.stride(1),
        dx.stride(0), dx.stride(1),
    )
    return dx


def main():
    print("=" * 60)
    print("16_weight_gradient — MatMul Backward (dW & dX)")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    sizes = [
        (256, 256, 128),
        (512, 512, 256),
        (1024, 1024, 512),
    ]

    for M, N, K in sizes:
        x = torch.randn(M, K, device="cuda", dtype=torch.float16)
        w = torch.randn(K, N, device="cuda", dtype=torch.float16)
        dy = torch.randn(M, N, device="cuda", dtype=torch.float16)

        # --- dW = X^T @ dY ---
        dw_triton = weight_gradient(x, dy)
        dw_ref = x.float().T @ dy.float()
        dw_max_diff = (dw_triton.float() - dw_ref).abs().max().item()
        dw_tol = max(0.01, 0.002 * (M ** 0.5))

        ms_dw_triton = do_bench(lambda: weight_gradient(x, dy))
        ms_dw_cublas = do_bench(lambda: x.T @ dy)  # cuBLAS
        flops_dw = (2 * M * N * K) / (ms_dw_triton * 1e-3) / 1e12

        # --- dX = dY @ W^T ---
        dx_triton = input_gradient(dy, w)
        dx_ref = dy.float() @ w.float().T
        dx_max_diff = (dx_triton.float() - dx_ref).abs().max().item()
        dx_tol = max(0.01, 0.002 * (N ** 0.5))

        ms_dx_triton = do_bench(lambda: input_gradient(dy, w))
        ms_dx_cublas = do_bench(lambda: dy @ w.T)  # cuBLAS
        flops_dx = (2 * M * N * K) / (ms_dx_triton * 1e-3) / 1e12

        dw_status = "✅" if dw_max_diff < dw_tol else "❌"
        dx_status = "✅" if dx_max_diff < dx_tol else "❌"
        print(f"  {M}×{K}×{N}:")
        print(f"    dW ({K}×{N}): Triton={ms_dw_triton:.4f}ms  cuBLAS={ms_dw_cublas:.4f}ms  "
              f"({ms_dw_cublas/ms_dw_triton:.2f}x)  "
              f"diff={dw_max_diff:.2e}  {dw_status}")
        print(f"    dX ({M}×{K}): Triton={ms_dx_triton:.4f}ms  cuBLAS={ms_dx_cublas:.4f}ms  "
              f"({ms_dx_cublas/ms_dx_triton:.2f}x)  "
              f"diff={dx_max_diff:.2e}  {dx_status}")


# PERFORMANCE NOTES
# =================
# - dW 和 dX 的计算量跟 forward 一样 (2*M*N*K)，但规约维度不同
# - dW = X^T @ dY: 规约 M 维（通常很大: batch * seq_len）
#   - arithmetic intensity 跟 forward 类似
# - dX = dY @ W^T: 规约 N 维（通常较小）
#   - arithmetic intensity 较低，更 memory-bound
# - 两者都需要处理隐式转置: 通过交换 offs 的维度顺序来避免显式转置
# - [COMPILER] 交换 offs 的维度顺序 → 生成不同的地址模式
#   - x_t [BK, BM]: 第一维是 K 偏移 → 线程在 K 维上相邻 → coalesced ✓
#   - w_t [BN, BK]: 第一维是 N 偏移 → 线程在 N 维上相邻 → coalesced ✓


if __name__ == "__main__":
    main()
