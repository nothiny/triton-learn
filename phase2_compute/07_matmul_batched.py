"""
14_matmul_batched.py — Batched GEMM: (B, M, K) @ (B, K, N) → (B, M, N)

学习目标:
  - 掌握 batched matmul 的 grid 设计（batch + M 折叠到 1D grid）
  - 理解为何 batched kernels 需要 stride 参数
  - 对比 triton 实现 vs torch.bmm vs for-loop matmul

Grid 设计:
  标准 matmul grid = (cdiv(M, BM), cdiv(N, BN))
  Batched: 把 B 和 M 维度折叠到 axis=0 → (B * cdiv(M, BM), cdiv(N, BN))
  在 kernel 内通过 pid_0 反推 (batch_id, pid_m)

运行: python phase2_compute/14_matmul_batched.py
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
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    ],
    key=["B", "M", "N", "K"],
)
@triton.jit
def batched_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    B, M, N, K,
    stride_ab, stride_am, stride_ak,  # A: (B, M, K) strides
    stride_bb, stride_bk, stride_bn,  # B: (B, K, N) strides
    stride_cb, stride_cm, stride_cn,  # C: (B, M, N) strides
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    (B, M, K) @ (B, K, N) → (B, M, N)
    """
    # Grid: axis0 = B * cdiv(M, BM), axis1 = cdiv(N, BN)
    pid_0 = tl.program_id(axis=0)
    pid_1 = tl.program_id(axis=1)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    # 从 1D pid_0 反推 (batch_id, pid_m)
    batch_id = pid_0 // num_pid_m
    pid_m = pid_0 % num_pid_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_1 * BLOCK_N + tl.arange(0, BLOCK_N)

    # 指针偏移到当前 batch
    a_ptr = a_ptr + batch_id * stride_ab
    b_ptr = b_ptr + batch_id * stride_bb
    c_ptr = c_ptr + batch_id * stride_cb

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def batched_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.dim() == 3 and b.dim() == 3
    B, M, K = a.shape
    B2, K2, N = b.shape
    assert B == B2 and K == K2

    c = torch.empty((B, M, N), device=a.device, dtype=a.dtype)

    grid = lambda meta: (
        B * triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    batched_matmul_kernel[grid](
        a, b, c,
        B, M, N, K,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        c.stride(0), c.stride(1), c.stride(2),
    )
    return c


def main():
    print("=" * 60)
    print("14_matmul_batched — Batched GEMM")
    print("=" * 60)

    configs = [
        (2, 256, 256, 256),
        (4, 512, 512, 512),
        (8, 512, 256, 512),
        (16, 1024, 1024, 1024),
    ]

    for B, M, N, K in configs:
        a = torch.randn(B, M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(B, K, N, device="cuda", dtype=torch.float16)

        c_triton = batched_matmul(a, b)
        c_ref = torch.bmm(a, b)

        max_diff = (c_triton.float() - c_ref.float()).abs().max().item()
        ms_triton = do_bench(lambda: batched_matmul(a, b))
        ms_cublas = do_bench(lambda: torch.bmm(a, b))
        tflops_triton = (2 * B * M * N * K) / (ms_triton * 1e-3) / 1e12
        tflops_cublas = (2 * B * M * N * K) / (ms_cublas * 1e-3) / 1e12

        status = "✅" if max_diff < 0.01 else "❌"
        print(f"  [{B}×{M}×{K}] @ [{B}×{K}×{N}]:")
        print(f"    Triton: {ms_triton:.4f}ms  {tflops_triton:.1f} TFLOPS")
        print(f"    cuBLAS: {ms_cublas:.4f}ms  {tflops_cublas:.1f} TFLOPS  "
              f"({ms_cublas/ms_triton:.2f}x vs Triton)  diff={max_diff:.2e}  {status}")


# PERFORMANCE NOTES
# =================
# - Batched GEMM 本质是把 batch 折叠到 grid 维度，kernel 内部用 stride 定位每 batch
# - 每个 batch 的矩阵独立: 不需要跨 batch 的同步或 shared memory 复用
# - 主要约束: shared memory 和 register 跟单 batch GEMM 一样
# - 如果 B 很大但 M/N 很小(如 B=64, M=N=64):
#   → 每个 batch 只产生 1 个 tile → grid 利用率低
#   → 更好的策略: 合并 batch 和 M 维度做 varlen 处理
# - 对比: torch.bmm 在内部也是类似实现，但 cuBLAS 提供了 strided batched GEMM API


if __name__ == "__main__":
    main()
