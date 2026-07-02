"""
19_cumsum.py — Parallel Prefix Sum (Cumulative Sum) kernel

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

运行: python phase1_fundamentals/19_cumsum.py
"""

import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.profiler import bench_compare, print_compare_report


@triton.jit
def _add(a, b):
    return a + b


@triton.jit
def cumsum_kernel(
    x_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Block 内并行 prefix sum。每个 thread 读 1 个元素，
    tl.associative_scan 在 shared memory 中用 Blelloch scan 完成 O(log N) 步。
    """
    pid = tl.program_id(axis=0)
    base = pid * BLOCK_SIZE
    offsets = base + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    val = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    result = tl.associative_scan(val, axis=0, combine_fn=_add)
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def add_carry_kernel(
    output_ptr, carry_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    把 block b 的 carry（= 前 b-1 个 block 的总和）加到 block b 的每个元素上。
    Block 0 不需要 carry。
    """
    pid = tl.program_id(axis=0)
    base = pid * BLOCK_SIZE
    offsets = base + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    val = tl.load(output_ptr + offsets, mask=mask, other=0.0)

    if pid > 0:
        carry = tl.load(carry_ptr + pid - 1)  # carry_ptr[b-1] = sum of blocks [0, b-1]
        val = val + carry

    tl.store(output_ptr + offsets, val, mask=mask)


def cumsum(x: torch.Tensor, BLOCK_SIZE: int = 1024) -> torch.Tensor:
    """
    完整 cumsum，自动处理跨 block carry 传播。

    Step 1: 每个 block 独立做 intra-block scan → output + block 总和
    Step 2: 对 block 总和做 cumsum → 得到每个 block 的 carry
    Step 3: 把 carry 加到对应 block 的每个元素
    """
    n = x.numel()
    n_blocks = triton.cdiv(n, BLOCK_SIZE)
    output = torch.empty_like(x)

    # Step 1: intra-block scan
    cumsum_kernel[(n_blocks,)](x, output, n, BLOCK_SIZE=BLOCK_SIZE)

    if n_blocks <= 1:
        return output

    # 提取每个 block 的总和（inclusive scan 的最后一个元素 = block sum）
    # output[BLOCK_SIZE-1::BLOCK_SIZE] 取每个 block 的最后一个元素
    # 但如果最后一个 block 不满，stride 可能漏掉 → 用 output[-1] 补上
    block_sums = output[BLOCK_SIZE - 1::BLOCK_SIZE].clone()
    if block_sums.numel() < n_blocks:
        block_sums = torch.cat([block_sums, output[-1].unsqueeze(0)])

    # 用 float64 算 carry，避免 float32 累积误差放大
    carry = torch.cumsum(block_sums.double(), dim=0).float()

    # Step 3: 把 carry 加到每个 block
    add_carry_kernel[(n_blocks,)](output, carry, n, BLOCK_SIZE=BLOCK_SIZE)

    return output


def main():
    print("=" * 60)
    print("19_cumsum — Parallel Prefix Sum (Cumsum)")
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
          f"{'✅' if max_diff < 5e-4 else '❌'}")

    # 正确性: 大 tensor (跨多个 BLOCK)
    x_large = torch.rand(4096, device="cuda")
    out_triton = cumsum(x_large)
    out_torch = torch.cumsum(x_large, dim=0)
    max_diff = (out_triton - out_torch).abs().max().item()
    print(f"  [multi-block ] max_diff={max_diff:.2e}  "
          f"{'✅' if max_diff < 5e-4 else '❌'}")

    print("\n--- Performance ---")
    x = torch.rand(65536, device="cuda", dtype=torch.float32)
    n = x.numel()
    result = bench_compare({
        "Triton (ours)": lambda: cumsum(x),
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
