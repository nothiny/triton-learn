"""
02_matmul_tiled.py — 生产级分块 GEMM（@triton.autotune 自动搜索最优配置）

学习目标：
  - 理解 @triton.autotune 的多维度搜索空间
  - 掌握 num_warps, num_stages 对性能的影响
  - 理解 software pipelining：用异步预取隐藏内存延迟
  - 学会做 roofline 分析：compute bound vs memory bound

运行: python phase2_compute/02_matmul_tiled.py
"""

import torch
import triton
import triton.language as tl
from triton.testing import do_bench


@triton.autotune(
    configs=[
        # 搜索空间: BLOCK_M × BLOCK_N × BLOCK_K × num_warps × num_stages
        # 原则: 覆盖 num_stages={2,3,4}, num_warps={2,4,8}, BLOCK_K={32,64,128}
        # 确保包含 01 的配置 (64,128,32) / num_warps=4 / num_stages=3

        # ---- 小 tile: BLOCK_M=64 ----
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4, num_stages=3),

        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),  # 01 的配置

        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4, num_stages=3),

        # ---- 中等 tile: BLOCK_M=128 ----
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),

        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),

        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=3),

        # ---- 大 tile: BLOCK_M=256 ----
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),

        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=4),

        # ---- 大 K block（减少 K 维迭代次数）----
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_tiled_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    生产级分块 MatMul: C[m, n] = sum_k A[m, k] * B[k, n]

    与 01 的 tl.load / tl.dot 写法完全相同。Triton 编译器自动负责数据放置
    （寄存器 / shared memory / L1），不需要手写 shared memory。

    与 01 的核心区别是 @triton.autotune:
    - autotune 对每个 (M,N,K) 在 8 组配置中逐组 benchmark，选最优的 BLOCK 尺寸
      + num_warps + num_stages 组合，结果按 key 缓存。
    - 01 只用一组硬编码配置 + 默认 num_warps=4, num_stages=3，
      无法针对不同矩阵形状自动调整。

    关于 num_stages:
    - num_stages 控制 software pipelining 的流水级数
    - 默认值是 3 (CUDAOptions.num_stages=3)
    - 01 隐式使用默认值 3；02 显式在 {2, 3} 中搜索最优
    - =2: double buffering, load[i+1] 与 compute[i] 重叠
    - =3: triple buffering, 更深的延迟隐藏（但占用更多 shared memory）
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        # Load A tile
        offs_k = k + tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        # [COMPILER] num_stages > 1 时，编译器将 a 放入 shared memory
        # 并插入 cp.async 做异步预取
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B tile
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # MMA: acc += a @ b
        acc += tl.dot(a, b)

    # Store C tile
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def matmul_tiled(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """生产级 MatMul wrapper: autotune 自动选择最优配置"""
    assert a.dim() == 2 and b.dim() == 2
    M, K = a.shape
    K2, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    matmul_tiled_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


def main():
    print("=" * 60)
    print("02_matmul_tiled — 生产级 GEMM（@triton.autotune）")
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
        c_triton = matmul_tiled(a, b)
        c_torch = torch.mm(a, b)
        max_diff = (c_triton.float() - c_torch.float()).abs().max().item()

        # 性能（triton.testing.do_bench: warmup 25ms + rep 100ms，取 mean）
        ms = do_bench(lambda: matmul_tiled(a, b))
        tflops = (2 * M * N * K) / (ms * 1e-3) / 1e12

        status = "✅" if max_diff < 0.01 else "❌"
        print(f"  {M}x{N}x{K}: {ms:.4f}ms  {tflops:.1f} TFLOPS  diff={max_diff:.2e}  {status}")


# PERFORMANCE NOTES
# =================
# - 本 kernel 的 tl.load / tl.dot 写法与 01_matmul_naive 完全相同。Triton 编译器
#   自动负责数据放置，不需要手写 shared memory。
#
# - 与 01 的真正区别（按重要性排序）:
#   1. @triton.autotune: 对每种 (M,N,K) 在 8 组配置中 benchmark 并选最优，
#      01 只用一组固定的 BLOCK 尺寸 (64×128×32)
#   2. autotune 搜索了 num_warps (4/8) 和 num_stages (2/3) 的多组组合，
#      而 01 用默认值 num_warps=4, num_stages=3
#   3. autotune 结果按 key=["M", "N", "K"] 缓存，同形状再次调用时直接复用
#
# - 两者都使用 software pipelining:
#   - 01: 隐式使用 CUDAOptions 默认 num_stages=3
#   - 02: autotune 在 {2, 3} 中搜索最优 num_stages
#
# - [COMPILER] 软件流水线 = VLIW 的 modulo scheduling:
#   - 编译器展开循环 → 重排指令 → 插入 cp.async prefetch
#   - num_stages=2: double buffering, 1 组 buffer 做 compute, 1 组做 load
#   - num_stages=3: triple buffering, 2 组 buffer 做 load (prefetch), 1 组做 compute
#   - 更多 stage 可以更好地隐藏延迟，但占用更多 shared memory
#
# - num_warps:
#   - 每个 warp = 32 threads，负责一部分 tile 的计算
#   - 更多 warp → 更高并行度，但每个 warp 分到的寄存器更少
#   - 默认 4 warps = 128 threads/block
#
# - Roofline 分析:
#   - H100 peak fp16: 989 TFLOPS, HBM: 3.35 TB/s
#   - Ridge point = 989e12 / 3.35e12 ≈ 295 FLOP/byte
#   - 对于大矩阵: 算术强度 ~ BLOCK/2 → 128/2 = 64 FLOP/byte < 295
#     → 理论上是 memory-bound
#   - 但 software pipelining 隐藏了访存延迟，实际性能可接近 compute-bound
#
# - 后续优化方向:
#   - Warp specialization (Hopper): producer warp 做 load, consumer warp 做 MMA
#   - FP8 数据类型: 2x 吞吐量
#   - Persistent kernel: 减少 grid launch 开销


if __name__ == "__main__":
    main()
