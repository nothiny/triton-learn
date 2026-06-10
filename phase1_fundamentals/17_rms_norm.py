"""
05_rms_norm.py — RMS Normalization kernel

学习目标：
  - 理解 RMSNorm vs LayerNorm 的差异（无 mean 居中，更快）
  - 掌握 2-pass reduction 模式（LayerNorm 是 3-pass）
  - 了解 RMSNorm 在现代 LLM 中的重要性（Llama, Mistral, Gemma 等）

数学公式:
  RMSNorm(x) = γ * x / √(mean(x²) + ε)
  相比 LayerNorm: 不做均值居中 (x - μ)，节省一次 reduction

关键差异:
  - LayerNorm: y = γ * (x - μ) / √(σ² + ε) + β  (3-pass: mean, var, norm)
  - RMSNorm:  y = γ * x / √(mean(x²) + ε)       (2-pass: sq_mean, norm)
  - RMSNorm 没有 bias 参数（只有 scale γ）
  - 经验上 RMSNorm 和 LayerNorm 效果相当，但快 ~10-15%

运行: python phase1_fundamentals/17_rms_norm.py
"""

import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.profiler import bench_compare, print_compare_report
from benchmarks.references.liger_ref import get_liger_rms_norm


@triton.jit
def rms_norm_kernel(
    x_ptr,        # 输入: (N_ROWS, N_COLS)
    weight_ptr,   # γ (scale), shape: (N_COLS,)
    output_ptr,   # 输出: (N_ROWS, N_COLS)
    n_cols,
    eps: tl.constexpr,         # epsilon (避免除零)
    BLOCK_SIZE: tl.constexpr,  # 每 program 处理的列数
):
    """
    逐行 RMSNorm: output[row, :] = γ * x[row, :] / √(mean(x²) + ε)

    每个 program 处理一行，2-pass 算法。
    """
    row_idx = tl.program_id(axis=0)  # 行索引
    row_start = row_idx * n_cols      # 当前行在展平内存中的起始偏移
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # ---- Pass 1: 计算 mean(x²) ----
    # RMS = Root Mean Square, 所以只需要平方的均值
    sq_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = row_start + block_start + col_offsets
        mask = (block_start + col_offsets) < n_cols
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        sq_sum += x * x

    # 全局 reduce: 将所有 block 的平方和汇总并求均值
    # tl.sum(sq_sum, axis=0): 跨 BLOCK_SIZE 个线程做 reduction
    mean_sq = tl.sum(sq_sum, axis=0) / n_cols

    # ---- Pass 2: 归一化并写回 ----
    # rsqrt(x) = 1 / sqrt(x), GPU 上有高效的硬件指令
    inv_rms = tl.math.rsqrt(mean_sq + eps)

    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = row_start + block_start + col_offsets
        mask = (block_start + col_offsets) < n_cols
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + block_start + col_offsets, mask=mask, other=0.0)

        # RMSNorm: γ * x / √(mean(x²) + ε) = γ * x * rsqrt(mean(x²) + ε)
        normalized = w * x * inv_rms
        tl.store(output_ptr + offsets, normalized, mask=mask)


def rms_norm(x: torch.Tensor, weight: torch.Tensor,
             eps: float = 1e-5) -> torch.Tensor:
    """
    RMSNorm 包装函数。

    Args:
        x: 输入张量 (N_ROWS, N_COLS)
        weight: scale 参数 γ (N_COLS,)
        eps: 数值稳定常数

    Returns:
        归一化后的张量 (N_ROWS, N_COLS)
    """
    output = torch.empty_like(x)
    n_rows, n_cols = x.shape
    grid = (n_rows,)  # 每行一个 program
    rms_norm_kernel[grid](x, weight, output, n_cols, eps=eps, BLOCK_SIZE=1024)
    return output


def main():
    print("=" * 60)
    print("05_rms_norm — Triton vs PyTorch vs Liger")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # ---- 正确性测试 ----
    N_ROWS, N_COLS = 2048, 4096
    torch.manual_seed(42)
    x = torch.randn(N_ROWS, N_COLS, device="cuda")
    weight = torch.randn(N_COLS, device="cuda")
    eps = 1e-5

    out_triton = rms_norm(x, weight, eps)

    # PyTorch 没有内置 RMSNorm，手动实现
    rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    out_torch = x * rms * weight

    max_diff = (out_triton - out_torch).abs().max().item()
    print(f"  Shape: ({N_ROWS}, {N_COLS})")
    print(f"  Max diff: {max_diff:.6e}")
    print(f"  Status: {'✅ PASS' if max_diff < 1e-3 else '❌ FAIL'}")

    # ---- 性能对比: Triton vs PyTorch vs Liger ----
    print("\n--- Performance ---")

    implementations = {
        "Triton (ours)": lambda: rms_norm(x, weight, eps),
        "PyTorch (ref)": lambda: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight,
    }

    # Add liger if available
    liger_rms = get_liger_rms_norm()
    if liger_rms:
        implementations["Liger (SotA)"] = lambda: liger_rms(x, weight, eps)

    n_elements = x.numel()
    # sq + mean + rsqrt + mul = ~5 FLOPs per element (比 LayerNorm ~8 少)
    flops_total = n_elements * 5
    # x(4B) + w(4B) + out(4B)
    bytes_total = n_elements * 2 * 4 + N_COLS * 4

    result = bench_compare(
        implementations,
        flops=flops_total,
        bytes_accessed=bytes_total,
        dtype="fp32",
    )
    print_compare_report(result)


# PERFORMANCE NOTES
# =================
# - RMSNorm 比 LayerNorm 快 ~10-15%: 少了 mean 计算 (2-pass vs 3-pass)
# - 也是 memory-bound: 算术强度 ≈ 2.5 FLOP/byte (LayerNorm ≈ 2.7)
# - rsqrt 是 GPU 硬件指令 (MUFU.SQRT + MUFU.RSQ), 非常快
# - 现代 LLM (Llama, Mistral, Gemma) 全部使用 RMSNorm 替代 LayerNorm
#   - 原因: 训练更快, 推理更快, 效果相当
# - 本实现是 2-pass (读 x 两次), 也可以做到 1-pass:
#   - block 内: 先读一批存 shared memory, 算好 mean_sq 后直接写回
#   - 代价: 需要 shared memory, 对长序列行可能不适用
# - [COMPILER] tl.math.rsqrt 编译为 PTX 的 rsqrt.approx.f32 指令
# - Liger 的 RMSNorm 是最优实现 (fused, 1-pass with shared memory)
#   - 和它对比较能看出优化空间

if __name__ == "__main__":
    main()
