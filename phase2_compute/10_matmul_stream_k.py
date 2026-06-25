"""
10_matmul_stream_k.py — Stream-K: Load-Balanced K-Dimension Parallelism

学习目标:
  - 理解 Stream-K 如何用动态 K tile 分配替代静态 split-K
  - 掌握 atomic counter 做 work dispatch 的模式
  - 理解 "先 K 后 MN" 的 tile 调度策略

背景:
  标准 GEMM: grid=(M tiles, N tiles), 每个 CTA 做全部 K → O(K) latency
  Split-K: grid=(M tiles, N tiles, SPLIT_K), 静态划分 K → load imbalance
  Stream-K: K tiles 是"流"入的，CTA 动态获取 K 范围

Stream-K 核心思想:
  每个 CTA 原子地获取一个 K tile 范围（如一组的 4 个 K tiles），
  对这组 K tiles 扫描所有 (M,N) tiles，做 partial matmul，
  结果通过 atomic_add 累加到 C。

  换组 K tiles → 继续扫描所有 (M,N) tiles → 重复。

  与 split-K 的区别:
    split-K: 每个 CTA 固定一对 (pid_m, pid_n)，K 静态划分
    stream-K: 每个 CTA 动态获取 K 范围，对所有 (M,N) 做贡献

运行: python phase2_compute/10_matmul_stream_k.py
"""

import torch
import triton
import triton.language as tl
from triton.testing import do_bench


@triton.jit
def stream_k_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    k_counter_ptr,       # 全局 atomic counter: 下一个 K tile block
    num_k_tiles,         # K 维的总 tile 数
    num_m_tiles,
    num_n_tiles,
    total_mn_tiles,      # = num_m_tiles * num_n_tiles
    tiles_per_k_group,   # 每个 CTA 一次获取多少个 K tile
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Stream-K GEMM: CTA 获取 K tile 范围 → 处理所有 (M,N) tiles → 重复。

    Grid: (num_sms,) — 固定的 CTA 池
    """
    # 持续获取 K 工作，直到所有 K tiles 被处理
    # Triton 不支持 while-break，用固定上限循环
    for _ in range(num_k_tiles):
        # 原子地获取一段 K tile 范围
        k_tile_start = tl.atomic_add(k_counter_ptr, tiles_per_k_group)

        if k_tile_start >= num_k_tiles:
            # 所有 K 工作已完成，后面不再做有用功
            pass
        else:
            k_tile_end = tl.minimum(k_tile_start + tiles_per_k_group, num_k_tiles)

            # 扫描所有 (M, N) tiles，对每个做 partial matmul
            for mn_idx in range(total_mn_tiles):
                pid_m = mn_idx // num_n_tiles
                pid_n = mn_idx % num_n_tiles

                offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
                offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

                acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

                # 处理获取到的 K 范围
                for k_tile in range(k_tile_start, k_tile_end):
                    k = k_tile * BLOCK_K
                    offs_k = k + tl.arange(0, BLOCK_K)

                    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
                    a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
                    a = tl.load(a_ptrs, mask=a_mask, other=0.0)

                    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
                    b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
                    b = tl.load(b_ptrs, mask=b_mask, other=0.0)

                    acc += tl.dot(a, b)

                # Atomic 累加到 C（多个 CTA 可能贡献到同一位置）
                c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
                c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
                tl.atomic_add(c_ptrs, acc, mask=c_mask)


def matmul_stream_k(a: torch.Tensor, b: torch.Tensor,
                    num_ctas=132, BLOCK_M=64, BLOCK_N=128, BLOCK_K=32,
                    tiles_per_k_group=4) -> torch.Tensor:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    c = torch.zeros((M, N), device=a.device, dtype=a.dtype)

    num_m_tiles = triton.cdiv(M, BLOCK_M)
    num_n_tiles = triton.cdiv(N, BLOCK_N)
    num_k_tiles = triton.cdiv(K, BLOCK_K)
    total_mn_tiles = num_m_tiles * num_n_tiles

    k_counter = torch.zeros(1, device=a.device, dtype=torch.int32)

    grid = (num_ctas,)

    stream_k_gemm_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        k_counter, num_k_tiles,
        num_m_tiles, num_n_tiles, total_mn_tiles,
        tiles_per_k_group,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c


def main():
    print("=" * 60)
    print("10_matmul_stream_k — Stream-K GEMM")
    print("=" * 60)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")
        num_sms = 132 if "H100" in gpu_name else 108

    configs = [
        (256, 256, 2048, 64, 64, 32, 4),
        (256, 256, 4096, 64, 64, 32, 4),
        (512, 512, 2048, 64, 128, 32, 4),
    ]

    for M, N, K, BM, BN, BK, k_group in configs:
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)

        c_triton = matmul_stream_k(a, b, num_sms, BM, BN, BK, k_group)
        c_ref = a @ b
        max_diff = (c_triton.float() - c_ref.float()).abs().max().item()
        tol = max(0.1, 0.02 * (K ** 0.5))  # atomic_add 引入较大累积误差

        ms_stream = do_bench(lambda: matmul_stream_k(a, b, num_sms, BM, BN, BK, k_group))
        ms_cublas = do_bench(lambda: a @ b)

        status = "✅" if max_diff < tol else "❌"
        print(f"  {M}×{K}×{N}: stream={ms_stream:.4f}ms  "
              f"cuBLAS={ms_cublas:.4f}ms  ({ms_cublas/ms_stream:.2f}x)  "
              f"diff={max_diff:.2e} (tol={tol:.1e})  {status}")


# PERFORMANCE NOTES
# =================
# - Stream-K 扫描所有 (M,N) tiles 对每个 K group，和标准 GEMM 的局部性相反
#   → 标准: 外循环 K, 内循环每 CTA 固定 (M,N)
#   → Stream-K: 外循环 (M,N), 内循环处理 K group
# - 这个实现是教学性质的，展示了动态 K 分配的概念
# - 真正的 Stream-K 实现会用更复杂的 tiling 来保持 cache 局部性
# - 参考: "Stream-K: Work-centric Parallel Decomposition for Dense
#   Matrix-Matrix Multiplication on the GPU" (Grelck et al., 2023)


if __name__ == "__main__":
    main()
