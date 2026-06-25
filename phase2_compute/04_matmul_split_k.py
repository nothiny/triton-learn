"""
04_matmul_split_k.py — Split-K 并行化 GEMM

学习目标:
  - 理解 Split-K 的并行策略：把 K 维拆分到多个 CTA
  - 掌握 atomic_add 做跨 CTA 归约
  - 理解什么时候用 split-K（K >> M,N 的场景）

Split-K 原理:
  标准 GEMM 的 grid 是 2D (M_tiles, N_tiles)，每个 CTA 顺序处理
  所有 K 个 block，累加到同一个 acc。当 K 很大时，单个 CTA 的
  SRAM 放不下完整 accumulator，或 K 迭代太多导致寄存器压力大。

  Split-K 把 K 维也并行化:
    - Grid: (M_tiles, N_tiles, SPLIT_K)
    - 每个 CTA 只处理 K/SPLIT_K 个 block
    - 每个 CTA 算出部分和 partial[m,n]
    - SPLIT_K 个 CTA 的 partial 通过 atomic_add 归约到最终结果

  代价:
    - atomic_add 有竞争开销
    - 并行度增加 → 更多 CTA → 更多 launch overhead
    - 每个 CTA 处理更少的 K → 更少的 arithmetic intensity

  最佳场景:
    - K 非常大 (如 8192+) 且 M,N 较小
    - 或者 register pressure 是瓶颈时
    - 不推荐: M,N 很大且 K 较小 (atomic_add 开销主导)

运行: python phase2_compute/04_matmul_split_k.py
"""

import torch
import triton
import triton.language as tl
from triton.testing import do_bench


@triton.jit
def matmul_split_k_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    K_BLOCKS_TOTAL,  # ceil(K / BLOCK_K)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """
    Split-K GEMM kernel (fixed tile sizes, no autotune).

    每个 CTA 负责 K 维的 1/SPLIT_K，使用 round-robin 分配。
    SPLIT_K 个 CTA 的 partial sums 通过 atomic_add 归约到 C。

    [COMPILER] SPLIT_K=1 时自动退化为标准 GEMM（tl.store）。
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_k = tl.program_id(axis=2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Round-robin K block 分配: pid_k 处理 blocks [pid_k, pid_k+SPLIT_K, ...]
    # [COMPILER] K_BLOCKS_TOTAL 是 kernel arg (Python int)，range 可正常迭代
    for k_block in range(pid_k, K_BLOCKS_TOTAL, SPLIT_K):
        k = k_block * BLOCK_K
        offs_k = k + tl.arange(0, BLOCK_K)

        # Load A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B tile: [BLOCK_K, BLOCK_N]
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

    # ---- 归约: store (SPLIT_K=1) 或 atomic_add (SPLIT_K>1) ----
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    if SPLIT_K == 1:
        tl.store(c_ptrs, acc, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, acc, mask=c_mask)


def matmul_split_k(
    a: torch.Tensor,
    b: torch.Tensor,
    split_k: int = 4,
) -> torch.Tensor:
    """
    Split-K GEMM.

    Args:
        a: (M, K) fp16/bf16
        b: (K, N) fp16/bf16
        split_k: K 维拆分份数。越大并行度越高，但 atomic_add 竞争也越大。
                 推荐: K >= 4096 时 split_k=4-8; K < 2048 时 split_k=1-2

    Returns:
        c: (M, N) same dtype as inputs
    """
    assert a.dim() == 2 and b.dim() == 2
    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    # 使用 fp32 中间缓冲做 atomic_add 累积
    c_fp32 = torch.zeros((M, N), device=a.device, dtype=torch.float32)

    # 固定 block 大小（学习用途，不做 autotune）
    BM, BN, BK = 128, 128, 32
    total_k_blocks = (K + BK - 1) // BK

    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN), split_k)

    matmul_split_k_kernel[grid](
        a, b, c_fp32,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c_fp32.stride(0), c_fp32.stride(1),
        total_k_blocks,
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        SPLIT_K=split_k,
    )
    return c_fp32.to(a.dtype)


# ==============================================================================
# Reference (standard tiled GEMM for comparison)
# ==============================================================================


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _standard_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Standard tiled GEMM (no split-K) for comparison."""
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
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


def matmul_standard(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Standard tiled GEMM for comparison."""
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    _standard_gemm_kernel[grid](a, b, c, M, N, K,
                                a.stride(0), a.stride(1),
                                b.stride(0), b.stride(1),
                                c.stride(0), c.stride(1))
    return c


# ==============================================================================
# Main
# ==============================================================================


def main():
    print("=" * 70)
    print("04_matmul_split_k — Split-K Parallel GEMM")
    print("=" * 70)

    torch.manual_seed(42)

    # Test configurations: vary K while keeping M,N fixed
    # Split-K shines when K >> M, N
    configs = [
        # (M, N, K, split_k, desc)
        (256, 256, 1024, 1, "K=1024, no split"),
        (256, 256, 1024, 4, "K=1024, split=4"),
        (256, 256, 4096, 1, "K=4096, no split"),
        (256, 256, 4096, 4, "K=4096, split=4"),
        (256, 256, 8192, 1, "K=8192, no split"),
        (256, 256, 8192, 8, "K=8192, split=8"),
        (1024, 1024, 4096, 1, "Large M,N, K=4096, no split"),
        (1024, 1024, 4096, 4, "Large M,N, K=4096, split=4"),
    ]

    for M, N, K, split_k, desc in configs:
        print(f"\n── {desc} ──")
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)

        # Split-K Triton
        c_split = matmul_split_k(a, b, split_k=split_k)

        # Reference: torch.mm
        c_ref = torch.mm(a.float(), b.float()).half()
        max_diff = (c_split.float() - c_ref.float()).abs().max().item()

        # Standard Triton GEMM (no split)
        c_std = matmul_standard(a, b)

        # Timing
        ms_split = do_bench(lambda: matmul_split_k(a, b, split_k=split_k))
        ms_std = do_bench(lambda: matmul_standard(a, b))

        tflops_split = (2 * M * N * K) / (ms_split * 1e-3) / 1e12
        tflops_std = (2 * M * N * K) / (ms_std * 1e-3) / 1e12

        # fp16 GEMM tolerance: ~0.05 * sqrt(K) due to different accumulation order
        tol = max(0.05, 0.005 * (K ** 0.5))
        status = "✅" if max_diff < tol else "❌"
        speedup = ms_std / ms_split
        print(f"  Standard: {ms_std:.4f}ms  {tflops_std:.1f} TFLOPS")
        print(f"  Split-K:  {ms_split:.4f}ms  {tflops_split:.1f} TFLOPS  "
              f"({speedup:.2f}x vs std)  diff={max_diff:.2e} (tol={tol:.1e})  {status}")

    # When to use split-K — analysis
    print(f"\n{'='*70}")
    print("Split-K Trade-off Analysis")
    print(f"{'='*70}")
    print("""
  Split-K 适用条件:
    1. K >> M, N (K 维度主导):         ✅ K=8192, M=N=256 → split 可能帮助
    2. K ≈ M ≈ N (立方矩阵):           ❌ 标准 GEMM 足够，split 增加 atomic 开销
    3. Register pressure 是瓶颈:       ✅ Split 减少 per-CTA 的 K 迭代次数
    4. SM 利用率不足 (小矩阵):          ✅ 3D grid 产生更多 CTA

  关键 trade-off:
    SPLIT_K ↑ : 更多 CTA → 更高 occupancy → 更多并行
    SPLIT_K ↑ : 更多 atomic_add 冲突 → 更高同步开销
    最佳 SPLIT_K = 2-8，取决于 K 的大小和 GPU 的 SM 数量

  [GPU] H100 有 132 个 SM，每个 SM 可同时跑多个 CTA。
  当 2D grid 的 CTA 数量 < SM 数量时，硬件利用不足。
  Split-K 通过增加第 3 维来产生更多 CTA，填满空闲 SM。
  """)


# PERFORMANCE NOTES
# =================
# - Split-K 的核心思想: 用 parallel reduction 换 synchronization
# - [COMPILER] SPLIT_K 是 tl.constexpr:
#   - 编译器按 SPLIT_K 展开 grid 的第 3 维
#   - 不同 SPLIT_K 需要重新编译 kernel
# - [GPU] atomic_add 的性能:
#   - 在同一 warp 内: 几乎无竞争（warp 内线程访问不同地址）
#   - 在同一 CTA 内: 通过 shared memory 仲裁
#   - 跨 CTA: L2 cache 原子操作，延迟 ~100 cycles
# - 与 cuBLAS split-K 的区别:
#   - cuBLAS 用额外的 reduction kernel 做归约（避免 atomics）
#   - Triton 版本用 atomic_add（实现简单，但在高竞争下有性能损失）
# - Roofline 分析:
#   - Split-K 降低每个 CTA 的 arithmetic intensity（处理更少的 K）
#   - 但通过更多 CTA 并行来提高整体吞吐
# - 后续学习:
#   - 对比 Split-K vs Stream-K（另一种 K 维并行策略）
#   - Stream-K 用 work-stealing 而非静态分配，更适合不规则 K


if __name__ == "__main__":
    main()
