"""
04_layer_norm.py — Layer Normalization 前向 kernel

学习目标：
  - 理解 LayerNorm 的数学和实现
  - 掌握 Welford 在线方差计算
  - 了解 shared memory reduction 的使用

数学公式:
  LayerNorm(x) = γ * (x - μ) / √(σ² + ε) + β
  其中 μ = mean(x), σ² = variance(x)

Welford 算法 (在线计算方差，无需两次遍历):
  M_0 = 0, S_0 = 0
  M_k = M_{k-1} + (x_k - M_{k-1}) / k
  S_k = S_{k-1} + (x_k - M_{k-1}) * (x_k - M_k)
  → μ = M_n, σ² = S_n / n

运行: python phase1_fundamentals/16_layer_norm.py
"""

import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.profiler import bench_compare, print_compare_report
from benchmarks.references.liger_ref import get_liger_ln


@triton.jit
def layer_norm_kernel(
    x_ptr,        # 输入: (N_ROWS, N_COLS)
    weight_ptr,   # γ (scale), shape: (N_COLS,)
    bias_ptr,     # β (shift), shape: (N_COLS,)
    output_ptr,   # 输出: (N_ROWS, N_COLS)
    n_cols,
    eps: tl.constexpr,         # epsilon (避免除零)
    BLOCK_SIZE: tl.constexpr,  # 每 program 处理的列数
):
    """
    逐行 LayerNorm: output[row, :] = γ * (x[row,:] - μ) / √(σ² + ε) + β

    每个 program 处理一行。
    """
    row_idx = tl.program_id(axis=0)
    row_start = row_idx * n_cols
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # ---- Step 1: 计算 mean（使用 Welford） ----
    mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = row_start + block_start + col_offsets
        mask = (block_start + col_offsets) < n_cols
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mean += x
    # 全局 mean（沿所有 BLOCK 平均）
    global_mean = tl.sum(mean, axis=0) / n_cols

    # ---- Step 2: 计算 variance ----
    variance = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = row_start + block_start + col_offsets
        mask = (block_start + col_offsets) < n_cols
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        diff = x - global_mean
        variance += diff * diff
    global_var = tl.sum(variance, axis=0) / n_cols

    # ---- Step 3: 归一化 + affine ----
    inv_std = 1.0 / tl.sqrt(global_var + eps)
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = row_start + block_start + col_offsets
        mask = (block_start + col_offsets) < n_cols
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + block_start + col_offsets, mask=mask, other=0.0)
        b = tl.load(bias_ptr + block_start + col_offsets, mask=mask, other=0.0)

        # LayerNorm: γ * (x - μ) / σ + β
        normalized = w * (x - global_mean) * inv_std + b
        tl.store(output_ptr + offsets, normalized, mask=mask)


def layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
               eps: float = 1e-5) -> torch.Tensor:
    output = torch.empty_like(x)
    n_rows, n_cols = x.shape
    grid = (n_rows,)
    layer_norm_kernel[grid](x, weight, bias, output, n_cols, eps=eps, BLOCK_SIZE=1024)
    return output


def main():
    print("=" * 60)
    print("04_layer_norm — Triton vs PyTorch")
    print("=" * 60)

    N_ROWS, N_COLS = 2048, 4096
    torch.manual_seed(42)
    x = torch.randn(N_ROWS, N_COLS, device="cuda")
    weight = torch.randn(N_COLS, device="cuda")
    bias = torch.randn(N_COLS, device="cuda")
    eps = 1e-5

    # 正确性
    out_triton = layer_norm(x, weight, bias, eps)
    out_torch = torch.nn.functional.layer_norm(x, [N_COLS], weight, bias, eps)

    max_diff = (out_triton - out_torch).abs().max().item()
    print(f"  Shape: ({N_ROWS}, {N_COLS})")
    print(f"  Max diff: {max_diff:.6e}")
    print(f"  Status: {'✅ PASS' if max_diff < 1e-3 else '❌ FAIL'}")

    # 性能对比: Triton vs PyTorch vs Liger
    print("\n--- Performance ---")

    implementations = {
        "Triton (ours)": lambda: layer_norm(x, weight, bias, eps),
        "PyTorch (ref)": lambda: torch.nn.functional.layer_norm(
            x, [N_COLS], weight, bias, eps=eps
        ),
    }

    # Add liger if available
    liger_ln = get_liger_ln()
    if liger_ln:
        implementations["Liger (SotA)"] = lambda: liger_ln(x, weight, bias, eps)

    n_elements = x.numel()
    flops_total = n_elements * 8  # mean + var + norm + affine ≈ 8 FLOPs per element
    bytes_total = n_elements * 3 * 4 + N_COLS * 2 * 4  # x + w + b read, out write

    result = bench_compare(
        implementations,
        flops=flops_total,
        bytes_accessed=bytes_total,
        dtype="fp32",
    )
    print_compare_report(result)


# PERFORMANCE NOTES
# =================
# - LayerNorm 是 memory-bound: 读输入 + 写输出，很少 FLOPs
# - 关键的优化: 将 mean 和 variance 计算融合进一个 pass（Welford 算法）
# - [COMPILER] tl.sum(x, axis=0) 会被编译为:
#     1. warp-level shuffle reduction (寄存器)
#     2. shared memory reduction (跨 warp)
# - 本实现是简化版（3 次读 x），完整版可以用 Welford 将 mean/var 合并为 1 次读


if __name__ == "__main__":
    main()
