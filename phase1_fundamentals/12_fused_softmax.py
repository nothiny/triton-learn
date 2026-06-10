"""
02_fused_softmax.py — Fused Softmax kernel

学习目标：
  - 掌握 Triton 的 reduction 操作 (tl.max, tl.sum)
  - 学会逐行 softmax 的分块实现
  - 理解为什么 fused softmax 比分离 kernel 快

数学公式:
  softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_i - max(x)))
  减去 max 是为了数值稳定性，避免 exp 溢出

运行: python phase1_fundamentals/12_fused_softmax.py
"""

import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.profiler import bench_compare, print_compare_report
from benchmarks.references.liger_ref import get_liger_softmax


@triton.jit
def softmax_kernel(
    x_ptr,       # 输入: (N_ROWS, N_COLS)
    output_ptr,  # 输出: (N_ROWS, N_COLS)
    n_cols,
    BLOCK_SIZE: tl.constexpr,  # 每 program 处理的列数
):
    """
    逐行 softmax: output[i, :] = softmax(input[i, :])

    每个 program 处理一行，沿列方向分块迭代。
    """
    row_idx = tl.program_id(axis=0)  # 行索引
    row_start = row_idx * n_cols     # 当前行在展平内存中的起始位置

    # 当前 program 在此迭代中处理的列偏移
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # ---- Step 1: 找每行的最大值（数值稳定性） ----
    # 如果一行超过 BLOCK_SIZE，需要迭代访问
    row_max = tl.full([BLOCK_SIZE], float("-inf"), dtype=tl.float32)
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = row_start + block_start + col_offsets
        mask = (block_start + col_offsets) < n_cols
        x = tl.load(x_ptr + offsets, mask=mask, other=float("-inf"))
        row_max = tl.maximum(row_max, x)

    # 全局 max: 将 row_max 中的所有值 reduction 到单个值
    global_max = tl.max(row_max, axis=0)

    # ---- Step 2: 计算 exp(x - max) 并累加 sum ----
    row_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = row_start + block_start + col_offsets
        mask = (block_start + col_offsets) < n_cols
        x = tl.load(x_ptr + offsets, mask=mask, other=float("-inf"))
        exp_x = tl.exp(x - global_max)
        row_sum += exp_x

    global_sum = tl.sum(row_sum, axis=0)

    # ---- Step 3: 归一化并写回 ----
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = row_start + block_start + col_offsets
        mask = (block_start + col_offsets) < n_cols
        x = tl.load(x_ptr + offsets, mask=mask, other=float("-inf"))
        softmax_val = tl.exp(x - global_max) / global_sum
        tl.store(output_ptr + offsets, softmax_val, mask=mask)


def fused_softmax(x: torch.Tensor) -> torch.Tensor:
    """行级 fused softmax"""
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)

    # grid: 每行一个 program
    grid = (n_rows,)
    softmax_kernel[grid](x, output, n_cols, BLOCK_SIZE=1024)
    return output


def main():
    print("=" * 60)
    print("02_fused_softmax — Triton vs PyTorch")
    print("=" * 60)

    # 测试正确性
    torch.manual_seed(42)
    N_ROWS, N_COLS = 1024, 4096
    x = torch.randn(N_ROWS, N_COLS, device="cuda")

    out_triton = fused_softmax(x)
    out_torch = torch.softmax(x, dim=-1)

    max_diff = (out_triton - out_torch).abs().max().item()
    print(f"  Shape: ({N_ROWS}, {N_COLS})")
    print(f"  Max diff: {max_diff:.6e}")
    print(f"  Status: {'✅ PASS' if max_diff < 1e-3 else '❌ FAIL'}")

    # 性能对比: Triton vs PyTorch vs Liger
    print("\n--- Performance ---")

    implementations = {
        "Triton (ours)": lambda: fused_softmax(x),
        "PyTorch (ref)": lambda: torch.softmax(x, dim=-1),
    }

    # Add liger if available
    liger_softmax = get_liger_softmax()
    if liger_softmax:
        implementations["Liger (SotA)"] = lambda: liger_softmax(x)

    n_elements = x.numel()
    flops_total = n_elements * 5  # exp + sub + div ≈ 5 FLOPs per element
    bytes_total = n_elements * 2 * 4  # read fp32 + write fp32

    result = bench_compare(
        implementations,
        flops=flops_total,
        bytes_accessed=bytes_total,
        dtype="fp32",
    )
    print_compare_report(result)


# PERFORMANCE NOTES
# =================
# - softmax 是 memory-bound: 读 1 次 float → 写 1 次 float，算术强度很低
# - Fused 的好处: 单 kernel 完成 3 次 pass (max → sum → normalize)
#   如果拆成 3 个 kernel: 每次都要 HBM round-trip，3x 内存带宽
#   Fused: 中间结果留在寄存器中，只读写各 1 次
# - [COMPILER] tl.max(x, axis=0) 会被编译为 warp shuffle + shared memory reduction
# - BLOCK_SIZE 应该 >= 行的典型长度，避免过多迭代


if __name__ == "__main__":
    main()
