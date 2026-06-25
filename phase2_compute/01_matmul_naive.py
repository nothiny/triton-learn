"""
01_matmul_naive.py — 最简单的分块 GEMM（固定参数，无 autotune）

学习目标：
  - 理解 GEMM 的 tiling 逻辑
  - 掌握 M/N/K 维度的分块策略
  - 学会计算 TFLOPS = 2 * M * N * K / time

运行: python phase2_compute/01_matmul_naive.py
"""

import torch
import triton
import triton.language as tl
from triton.testing import do_bench


@triton.jit
def matmul_naive_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,  # A 的 stride (M, K)
    stride_bk, stride_bn,  # B 的 stride (K, N)
    stride_cm, stride_cn,  # C 的 stride (M, N)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    基础分块 MatMul: C[m, n] = sum_k A[m, k] * B[k, n]

    Triton 编译器自动负责数据放置（寄存器 / shared memory / L1），不需要手写
    shared memory。未指定 num_stages 时使用默认值 3（triple buffering 软件流水线），
    编译器会自动插入 cp.async 做异步预取。

    每个 program 负责计算 C 的一个 [BLOCK_M × BLOCK_N] tile。
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # 累加器: [BLOCK_M, BLOCK_N]，保持在寄存器中
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # 沿 K 维迭代（K 可能远超 BLOCK_K）
    for k in range(0, K, BLOCK_K):
        # A tile: [BLOCK_M, BLOCK_K]
        offs_k = k + tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # B tile: [BLOCK_K, BLOCK_N]
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # MMA: acc += A_tile @ B_tile
        acc += tl.dot(a, b)

    # 写回 C
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def matmul_naive(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    基础 MatMul wrapper: 硬编码一组 BLOCK 尺寸，使用 Triton 默认的
    num_stages=3, num_warps=4。无 autotune。
    """
    assert a.dim() == 2 and b.dim() == 2
    assert a.shape[1] == b.shape[0], f"dim mismatch: {a.shape} @ {b.shape}"

    M, K = a.shape
    K2, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    matmul_naive_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=64, BLOCK_N=128, BLOCK_K=32,
    )
    return c


def main():
    print("=" * 60)
    print("01_matmul_naive — 基础分块 GEMM（固定参数，无 autotune）")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    sizes = [
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
    ]

    for M, N, K in sizes:
        a = torch.randn((M, K), device="cuda", dtype=torch.float16)
        b = torch.randn((K, N), device="cuda", dtype=torch.float16)

        # 正确性
        c_triton = matmul_naive(a, b)
        c_torch = torch.mm(a, b)
        max_diff = (c_triton.float() - c_torch.float()).abs().max().item()

        # 性能（triton.testing.do_bench: warmup 25ms + rep 100ms，取 mean）
        ms = do_bench(lambda: matmul_naive(a, b))
        tflops = (2 * M * N * K) / (ms * 1e-3) / 1e12

        status = "✅" if max_diff < 0.01 else "❌"
        print(f"  {M}x{N}x{K}: {ms:.4f}ms  {tflops:.1f} TFLOPS  diff={max_diff:.2e}  {status}")


# PERFORMANCE NOTES
# =================
# - 本 kernel 的 tl.load / tl.dot 写法与 02_matmul_tiled 完全相同。Triton 编译器
#   自动决定数据放置（寄存器 / shared memory / L1），不需要手写 shared memory。
#
# - 与 02_matmul_tiled 的真正区别（按重要性排序）:
#   1. 没有 @triton.autotune: 只用一组固定的 BLOCK 尺寸 (64×128×32)，
#      无法针对不同 (M,N,K) 自动选择最优配置
#   2. num_warps 和 num_stages 用默认值 (4, 3)，而不是在搜索空间中尝试
#      多组组合（02 尝试 num_warps=4/8, num_stages=2/3）
#
# - 两者都使用 software pipelining:
#   - 01: 隐式使用默认 num_stages=3 (triple buffering)
#   - 02: autotune 在 {2, 3} 中选择最优 num_stages
#   - CUDAOptions 默认: num_stages=3, num_warps=4
#     (见 triton/backends/nvidia/compiler.py:106)
#
# - 算术强度: (2*M*N*K) FLOP / ((M*K + K*N + M*N) * dtype_size) bytes
#   - 对于大矩阵: 趋于 O(N) FLOP / O(N²) bytes → 随尺寸增长而提高
#
# - 主要瓶颈: 固定 block size 无法适配所有矩阵形状（例如 tall-skinny 矩阵
#   需要不同的 BLOCK_M/BLOCK_N 比例）
#
# - 下一步: 02_matmul_tiled.py 加入 @autotune，自动搜索最优配置
#
# - [COMPILER] tl.dot(a, b) 自动映射为 MMA 指令。Triton 根据 BLOCK 尺寸
#   选择合适的 MMA 布局 (MmaEncodingAttr v1/v2/v3)


if __name__ == "__main__":
    main()
