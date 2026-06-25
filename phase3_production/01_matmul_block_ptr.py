"""
phase3_production/01_matmul_block_ptr.py — 生产级 GEMM (Block Pointer API)

对比 phase2_compute/02_matmul_tiled.py，学习:
  - tl.make_block_ptr 替代手工指针拼接
  - boundary_check 替代手工 mask
  - 编译器如何利用 block pointer 做更好的 coalescing / prefetch
  - Hopper TMA (Tensor Memory Access) 的前置知识

Block Pointer API (Triton 2.1+):
  - 描述数据的形状/stride/offset/block_shape，编译器负责生成地址
  - tl.load(p_block, boundary_check=(0, 1)) 自动处理边界 mask
  - 编译器可以推理访问模式 → 更好的指令调度和 prefetch
  - 在 Hopper (SM90) 上可以映射为 TMA 指令

运行: python phase3_production/01_matmul_block_ptr.py
"""

import torch
import triton
import triton.language as tl
from triton.testing import do_bench


@triton.autotune(
    configs=[
        triton.Config({"BM": 64, "BN": 64, "BK": 32}, num_warps=4, num_stages=2),
        triton.Config({"BM": 64, "BN": 128, "BK": 32}, num_warps=4, num_stages=3),
        triton.Config({"BM": 128, "BN": 64, "BK": 32}, num_warps=4, num_stages=3),
        triton.Config({"BM": 128, "BN": 128, "BK": 32}, num_warps=8, num_stages=2),
        triton.Config({"BM": 128, "BN": 128, "BK": 64}, num_warps=8, num_stages=3),
        triton.Config({"BM": 128, "BN": 256, "BK": 32}, num_warps=8, num_stages=3),
        triton.Config({"BM": 256, "BN": 128, "BK": 32}, num_warps=8, num_stages=3),
        triton.Config({"BM": 256, "BN": 256, "BK": 64}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_block_ptr_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """
    C = A @ B，使用 tl.make_block_ptr。

    对比 02_matmul_tiled.py 的老写法:
      老: offs_m = pid_m * BM + tl.arange(0, BM)
          a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
          a = tl.load(a_ptrs, mask=mask, other=0.0)
      新: p_a = tl.make_block_ptr(base=a_ptr, shape=(M,K), strides=(sa_m,sa_k),
                                   offsets=(pid_m*BM, 0), block_shape=(BM,BK), order=(1,0))
          a = tl.load(p_a, boundary_check=(0,1))
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # ── Block pointers ──────────────────────────────────────────
    # make_block_ptr(base, shape, strides, offsets, block_shape, order)
    # order=(1,0): 第一维是 leading dim (stride=1 的维度先迭代)

    # A tile: [BM, BK]，从 A[pid_m*BM:, :] 切片，K 维循环推进
    p_a = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BM, 0),      # 初始 K 偏移=0，循环中通过 advance 更新
        block_shape=(BM, BK),
        order=(1, 0),                 # K 维 stride=1 → order[0]=1
    )

    # B tile: [BK, BN]
    p_b = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BN),      # 初始 K 偏移=0
        block_shape=(BK, BN),
        order=(1, 0),
    )

    acc = tl.zeros([BM, BN], dtype=tl.float32)

    # 沿 K 维循环: advance(p_a, (0, BK)) 推进 B 的 K 偏移 BK
    for k in range(0, K, BK):
        a = tl.load(p_a, boundary_check=(0, 1))  # 自动 mask: M 维和 K 维
        b = tl.load(p_b, boundary_check=(0, 1))

        acc += tl.dot(a, b)

        # advance(ptr, offsets): 同时推进两个 pointer 的 K 维
        p_a = tl.advance(p_a, (0, BK))
        p_b = tl.advance(p_b, (BK, 0))

    # C tile: [BM, BN]
    p_c = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BM, pid_n * BN),
        block_shape=(BM, BN),
        order=(1, 0),
    )
    tl.store(p_c, acc.to(tl.float16), boundary_check=(0, 1))


def matmul_block_ptr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta["BM"]),
        triton.cdiv(N, meta["BN"]),
    )

    matmul_block_ptr_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


def main():
    print("=" * 60)
    print("01_matmul_block_ptr — Production GEMM with Block Pointer API")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    sizes = [
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ]

    for M, N, K in sizes:
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)

        c_triton = matmul_block_ptr(a, b)
        c_ref = a @ b
        max_diff = (c_triton.float() - c_ref.float()).abs().max().item()

        ms_triton = do_bench(lambda: matmul_block_ptr(a, b))
        ms_cublas = do_bench(lambda: a @ b)
        tflops_t = (2 * M * N * K) / (ms_triton * 1e-3) / 1e12
        tflops_c = (2 * M * N * K) / (ms_cublas * 1e-3) / 1e12

        status = "✅" if max_diff < 0.01 else "❌"
        print(f"  {M}×{K}×{N}: Triton={ms_triton:.4f}ms ({tflops_t:.0f} TF)  "
              f"cuBLAS={ms_cublas:.4f}ms ({tflops_c:.0f} TF)  "
              f"({ms_cublas/ms_triton:.2f}x)  diff={max_diff:.2e}  {status}")


# PERFORMANCE NOTES
# =================
# tl.make_block_ptr vs 手工指针拼接:
#
# 1. 代码简洁性:
#    老: 3 行（offs + ptrs + mask） → 新: 1 行（boundary_check）
#
# 2. 编译器优化:
#    block pointer 暴露了编译时已知的 shape 和 stride，
#    编译器可以:
#    - 推断 coalesced access pattern
#    - 插入更好的 prefetch (cp.async)
#    - 减少地址计算指令
#
# 3. TMA (Hopper SM90+):
#    tl.make_block_ptr 在 Hopper 上可以映射为 TMA 指令:
#    - cp.async.bulk: 硬件级异步拷贝，零寄存器开销
#    - TMA descriptor: 硬件管理地址计算和边界检查
#    这需要 order=(0,1) 或 (1,0) 匹配 TMA 要求
#
# 4. 性能:
#    对中等规模 GEMM，block_ptr 版本的性能与手工指针拼接接近
#    （因为底层的 MMA 才是瓶颈）。差异主要体现在:
#    - 减少地址计算指令 → 更少寄存器压力
#    - 更好的 prefetch 调度 → 略高的 bandwidth utilization


if __name__ == "__main__":
    main()
