"""
02_layout_analysis.py — 分析 Triton 的 layout encoding

学习目标：
  - 理解 BlockedEncodingAttr 的参数含义
  - 可视化 thread → element 的映射关系
  - 了解不同 layout 之间的转换开销

运行: python phase3_compiler/02_layout_analysis.py
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# 辅助函数：解析和可视化 layout
# ---------------------------------------------------------------------------


def explain_blocked_encoding(
    size_per_thread: tuple,
    threads_per_warp: tuple,
    warps_per_cta: tuple,
    order: tuple,
) -> str:
    """
    Explain what a BlockedEncodingAttr means.

    BlockedEncodingAttr describes how a tensor shape is distributed:
      - First, the shape is divided among warps (by warps_per_cta)
      - Within each warp, the shape is divided among threads (by threads_per_warp)
      - Each thread holds size_per_thread elements

    The 'order' parameter determines which dimension is "innermost"
    (contiguous in memory), similar to row-major vs column-major.
    """
    lines = []
    lines.append("BlockedEncodingAttr{")
    lines.append(f"  sizePerThread  = {size_per_thread}")
    lines.append(f"  threadsPerWarp = {threads_per_warp}")
    lines.append(f"  warpsPerCTA    = {warps_per_cta}")
    lines.append(f"  order          = {order}")
    lines.append("}")

    # Compute total elements per CTA
    total_elements = 1
    for i in range(len(size_per_thread)):
        total_elements *= size_per_thread[i] * threads_per_warp[i] * warps_per_cta[i]
    lines.append(f"\n  Total elements per CTA: {total_elements}")

    # Explain thread→element mapping
    lines.append("\n  Element mapping: element[W, T, E] = ...")
    lines.append(f"    where W = warp index ({warps_per_cta})")
    lines.append(f"          T = thread index ({threads_per_warp})")
    lines.append(f"          E = element index per thread ({size_per_thread})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 创建不同 layout 的 kernel 用于分析
# ---------------------------------------------------------------------------


@triton.jit
def kernel_blocked_layout(
    x_ptr, y_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Simple kernel to observe BlockedEncodingAttr.
    The compiler will apply a blocked layout to the tensors.
    """
    pid = tl.program_id(axis=0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # This 2D tensor will get a BlockedEncodingAttr
    x = tl.load(x_ptr + offs_m[:, None] * BLOCK_N + offs_n[None, :])
    y = tl.load(y_ptr + offs_m[:, None] * BLOCK_N + offs_n[None, :])
    result = x + y  # elementwise: same layout as input

    # [COMPILER] Triton chooses the layout automatically:
    # - Default: BlockedEncodingAttr with sizePerThread determined by heuristics
    # - For dot products: may use MmaEncodingAttr


@triton.jit
def kernel_with_reduce(
    x_ptr, y_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Kernel with reduction to observe SliceEncodingAttr.
    """
    pid = tl.program_id(axis=0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    x = tl.load(x_ptr + offs_m[:, None] * BLOCK_N + offs_n[None, :])

    # Reduction along axis=1 creates SliceEncodingAttr
    # [COMPILER] The reduced dimension gets a different layout
    row_sum = tl.sum(x, axis=1)

    tl.store(y_ptr + offs_m, row_sum)


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("02_layout_analysis — Layout Encoding Deep Dive")
    print("=" * 60)

    # 解释一个典型的 BlockedEncodingAttr
    print("\n[1] Typical BlockedEncodingAttr for a vector (1D):")
    print(explain_blocked_encoding(
        size_per_thread=(1,),
        threads_per_warp=(32,),
        warps_per_cta=(4,),
        order=(0,),
    ))

    print("\n[2] Typical BlockedEncodingAttr for a 2D tile (e.g., GEMM):")
    print(explain_blocked_encoding(
        size_per_thread=(1, 4),     # each thread: 1 element in dim 0, 4 in dim 1
        threads_per_warp=(2, 16),   # 2×16=32 threads per warp
        warps_per_cta=(4, 1),       # 4 warps along dim 0, 1 warp along dim 1
        order=(0, 1),               # dim 0 innermost → row-major
    ))

    print("\n[3] Layout types summary:")
    print("""
  BlockedEncodingAttr  → standard blocked layout, used for elementwise ops
  SliceEncodingAttr    → for reduction along one dimension
  MmaEncodingAttr      → Tensor Core MMA layout (warp-level)
  DotOperandEncodingAttr → operands to tl.dot (A/B sides)
  ScanEncodingAttr     → for scan/prefix-sum operations
""")

    # Run a kernel to see what layout Triton chooses
    print("[4] Running kernel to trigger compilation...")
    M, N = 64, 128
    x = torch.randn(M, N, device="cuda")
    y = torch.randn(M, N, device="cuda")
    z = torch.empty(M, N, device="cuda")
    sums = torch.empty(M, device="cuda")

    kernel_blocked_layout[(1,)](x, y, BLOCK_M=64, BLOCK_N=128)
    kernel_with_reduce[(1,)](x, sums, BLOCK_M=64, BLOCK_N=128)
    torch.cuda.synchronize()

    print("  Done. Check ~/.triton/cache/ for .ttgir files with layout annotations.")
    print("  Look for: #blocked<{...}>, #slice<{...}>, etc.")


if __name__ == "__main__":
    main()
