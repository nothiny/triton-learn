"""
11_matmul_persistent.py — Persistent Kernel GEMM

学习目标:
  - 理解 GPU kernel launch overhead
  - 掌握 persistent kernel 设计模式（固定 grid + 动态 work dispatch）
  - 学习 atomic counter 做 work stealing 的模式

背景:
  标准 GEMM grid = cdiv(M,BM) * cdiv(N,BN) 个 CTA
  当 M,N 很小时 → CTA 数 < SM 数 → 部分 SM 空闲

  Persistent kernel: 只启动 num_sms 个 CTA，每个 CTA 循环获取工作。
  通过 atomic counter 分配 (pid_m, pid_n) tile。

运行: python phase2_compute/11_matmul_persistent.py
"""

import torch
import triton
import triton.language as tl
from triton.testing import do_bench


@triton.jit
def persistent_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    work_counter_ptr,    # 全局 atomic counter
    total_mn_tiles,      # = num_m_tiles * num_n_tiles
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Persistent GEMM: 每个 CTA 循环获取 (pid_m, pid_n) 并处理对应 tile。

    使用 for 循环（上限 = total_mn_tiles）而非 while-break，
    因为 Triton 不支持 break。
    """
    num_n_tiles = tl.cdiv(N, BLOCK_N)

    # 最坏情况下一个 CTA 做所有 tile → 循环 total_mn_tiles 次
    for _ in range(total_mn_tiles):
        tile_idx = tl.atomic_add(work_counter_ptr, 1)

        if tile_idx < total_mn_tiles:
            pid_m = tile_idx // num_n_tiles
            pid_n = tile_idx % num_n_tiles

            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

            acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

            for k in range(0, K, BLOCK_K):
                offs_k = k + tl.arange(0, BLOCK_K)

                a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
                a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
                a = tl.load(a_ptrs, mask=a_mask, other=0.0)

                b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
                b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
                b = tl.load(b_ptrs, mask=b_mask, other=0.0)

                acc += tl.dot(a, b)

            c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
            tl.store(c_ptrs, acc, mask=c_mask)


def matmul_persistent(a: torch.Tensor, b: torch.Tensor,
                      num_ctas=132, BLOCK_M=64, BLOCK_N=128, BLOCK_K=32) -> torch.Tensor:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    num_m_tiles = triton.cdiv(M, BLOCK_M)
    num_n_tiles = triton.cdiv(N, BLOCK_N)
    total_mn_tiles = num_m_tiles * num_n_tiles

    work_counter = torch.zeros(1, device=a.device, dtype=torch.int32)

    grid = (num_ctas,)

    persistent_gemm_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        work_counter, total_mn_tiles,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c


def main():
    print("=" * 60)
    print("11_matmul_persistent — Persistent Kernel GEMM")
    print("=" * 60)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")
        num_sms = 132 if "H100" in gpu_name else 108
    else:
        num_sms = 108

    configs = [
        (64, 64, 256, 64, 64, 32),        # 小矩阵 — persistent 优势
        (128, 128, 256, 64, 64, 32),      # 小矩阵
        (256, 256, 512, 64, 128, 32),     # 中等
        (512, 512, 512, 64, 128, 64),     # 较大 — 标准 launch 更好
        (1024, 1024, 1024, 128, 128, 64), # 大 — 验证正确性
    ]

    for M, N, K, BM, BN, BK in configs:
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)

        c_triton = matmul_persistent(a, b, num_sms, BM, BN, BK)
        c_ref = a @ b
        max_diff = (c_triton.float() - c_ref.float()).abs().max().item()

        ms_persist = do_bench(lambda: matmul_persistent(a, b, num_sms, BM, BN, BK))
        ms_cublas = do_bench(lambda: a @ b)

        num_tiles = triton.cdiv(M, BM) * triton.cdiv(N, BN)
        grid_util = min(1.0, num_tiles / num_sms)
        tflops_triton = (2 * M * N * K) / (ms_persist * 1e-3) / 1e12
        tflops_cublas = (2 * M * N * K) / (ms_cublas * 1e-3) / 1e12

        status = "✅" if max_diff < 0.01 else "❌"
        print(f"  {M}×{K}×{N}, tiles={num_tiles} (grid util={grid_util:.1%}):")
        print(f"    persistent={ms_persist:.4f}ms ({tflops_triton:.1f} TF)  "
              f"cuBLAS={ms_cublas:.4f}ms ({tflops_cublas:.1f} TF)  "
              f"({ms_cublas/ms_persist:.2f}x)  diff={max_diff:.2e}  {status}")


# PERFORMANCE NOTES
# =================
# - 每个 CTA 循环 for _ in range(total_mn_tiles): atomic 获取 + if 条件处理
#   - 当 tile_idx >= total_mn_tiles 时，CTA 进入 if 的 else（空转）
#   - 但 CTA 不提前退出（Triton 不支持 break），所有 CTA 循环 total_mn_tiles 次
# - 小矩阵优势:
#   - tiles < num_sms: 标准 launch 会让部分 SM 空闲
#   - persistent: 每个 SM 持续处理 tile 直到完成
# - 大矩阵: 标准 launch 更好（每个 CTA 一个 tile，无 atomic 竞争）
# - launch overhead: ~5-10 μs per kernel call
# - atomic counter: ~1 cycle（warp-level atomic）


if __name__ == "__main__":
    main()
