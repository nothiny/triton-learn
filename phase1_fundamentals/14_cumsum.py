"""
14_cumsum.py — Parallel Prefix Sum (Cumulative Sum) kernel

学习目标：
  - 理解并行 scan 算法 (Blelloch scan / Hillis-Steele scan)
  - 掌握 warp shuffle 和 shared memory 的协同使用
  - 这是 Flash Attention、sort、radix select 等高级 kernel 的基础原语

算法原理 (Blelloch scan, 以 8 个元素为例):

  上扫 (Up-sweep / Reduce):
    step 1: [a, a+b, c, c+d, e, e+f, g, g+h]
    step 2: [a, a+b, c, a+b+c+d, e, e+f, g, e+f+g+h]
    step 3: [a, a+b, c, a+b+c+d, e, e+f, g, a+b+c+d+e+f+g+h]

  下扫 (Down-sweep / Propagate):
    step 1: [0, a, a+b, c, a+b+c+d, e, e+f, g]
    step 2: [0, a, a+b, a+b+c, a+b+c+d, a+b+c+d+e, e+f, a+b+c+d+e+f]
    step 3: [0, a, a+b, a+b+c, a+b+c+d, a+b+c+d+e, a+b+c+d+e+f, a+b+...+g]

  输出: 每个位置是之前所有元素的和

为什么重要:
  - cumsum 是很多算法的 building block
  - Flash Attention 中 online softmax 依赖 local cumsum 维护 running sum
  - Triton 有 tl.associative_scan 可以直接用, 但理解原理很重要

运行: python phase1_fundamentals/14_cumsum.py
"""

import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.profiler import bench_compare, print_compare_report


@triton.jit
def cumsum_kernel(
    x_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Block-level prefix sum: 每个 program 在 BLOCK_SIZE 内做顺序 cumsum.

    这是教学简化版 (O(BLOCK_SIZE) per thread).
    生产级实现用 parallel scan (log-depth).
    """
    pid = tl.program_id(axis=0)
    base = pid * BLOCK_SIZE

    # 顺序 cumsum within block
    accum = 0.0
    for i in range(BLOCK_SIZE):
        idx = base + i
        if idx < n_elements:
            val = tl.load(x_ptr + idx)
            accum = accum + val
            tl.store(output_ptr + idx, accum)


def cumsum(x: torch.Tensor) -> torch.Tensor:
    """
    Block-level cumulative sum.

    注意: 此简化版仅在 BLOCK_SIZE 内做 cumsum.
    完整实现需要处理跨 block 传播 (add last element of prev block to all of current).
    """
    output = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    cumsum_kernel[grid](x, output, n, BLOCK_SIZE=1024)
    return output


def cumsum_full(x: torch.Tensor, BLOCK_SIZE: int = 1024) -> torch.Tensor:
    """
    完整 cumsum: 处理跨 block 的 carry-over.
    Step 1: block 内 cumsum (每个 block 独立)
    Step 2: 提取每个 block 的最后一个值, 做 cumsum 得到 block_carry
    Step 3: 将 carry 加到后续 block
    """
    n = x.numel()
    n_blocks = triton.cdiv(n, BLOCK_SIZE)
    output = torch.empty_like(x)

    # Step 1: block 内 cumsum
    grid = (n_blocks,)
    cumsum_kernel[grid](x, output, n, BLOCK_SIZE=BLOCK_SIZE)

    # Step 2 & 3: cross-block carry propagation
    # 提取每个 block 的最后一个有效元素
    last_vals = torch.zeros(n_blocks, device=x.device, dtype=x.dtype)
    for b in range(n_blocks):
        end = min((b + 1) * BLOCK_SIZE, n)
        if end > b * BLOCK_SIZE:
            last_vals[b] = output[end - 1]

    # 对 last_vals 做 cumsum (递归), 得到每个 block 需要加的 carry
    if n_blocks > 1:
        block_carry = torch.cumsum(last_vals, dim=0)
        # 每个 block b 加上前一个 block 的 carry
        for b in range(1, n_blocks):
            start = b * BLOCK_SIZE
            end = min((b + 1) * BLOCK_SIZE, n)
            output[start:end] += block_carry[b - 1]

    return output


def main():
    print("=" * 60)
    print("14_cumsum — Parallel Prefix Sum (Cumsum)")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    torch.manual_seed(42)

    # 正确性: 小 tensor (在 1 个 BLOCK 内)
    x_small = torch.rand(128, device="cuda")
    out_triton = cumsum(x_small)
    out_torch = torch.cumsum(x_small, dim=0)
    max_diff = (out_triton - out_torch).abs().max().item()
    print(f"  [single-block] max_diff={max_diff:.2e}  "
          f"{'✅' if max_diff < 1e-5 else '❌'}")

    # 正确性: 大 tensor (跨多个 BLOCK, 完整版)
    x_large = torch.rand(4096, device="cuda")
    out_full = cumsum_full(x_large)
    out_torch_full = torch.cumsum(x_large, dim=0)
    max_diff_full = (out_full - out_torch_full).abs().max().item()
    print(f"  [multi-block ] max_diff={max_diff_full:.2e}  "
          f"{'✅' if max_diff_full < 1e-4 else '❌'}")

    print("\n--- Performance ---")
    x = torch.rand(65536, device="cuda", dtype=torch.float32)
    n = x.numel()
    result = bench_compare({
        "Triton (ours)": lambda: cumsum_full(x),
        "PyTorch (ref)": lambda: torch.cumsum(x, dim=0),
    }, flops=n * 1, bytes_accessed=n * 2 * 4, dtype="fp32")
    print_compare_report(result)


# PERFORMANCE NOTES
# =================
# - tl.associative_scan 是 Triton 的高级原语:
#   - 内部使用 warp shuffle (寄存器级) + shared memory 实现
#   - 复杂度 O(log N) 步, 每步 O(N) 工作 → O(N log N) 总工作量
#   - 对 BLOCK_SIZE=1024: ~10 步 (log₂ 1024)
# - 跨 block cumsum 需要 carry propagation (见 cumsum_full)
# - 应用场景:
#   - Flash Attention: online softmax 的 running sum
#   - Ray tracing: 区间树构建
#   - Sorting & selection: radix sort 依赖 scan
# - PyTorch 的 cumsum 使用 CUB (CUDA UnBound) 库, 高度优化

if __name__ == "__main__":
    main()
