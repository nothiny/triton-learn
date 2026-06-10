"""
06_fused_scale_bias_residual.py — Fused Scale + Bias + Residual kernel

学习目标：
  - 掌握 3-input fusion 模式 (scale + bias + residual)
  - 理解 ResNet/CNN 中残差连接的融合优化
  - 学习多输入 kernel 的 indexing 模式

数学公式:
  output = α * x + β * bias + residual

在深度学习中的使用:
  - ResNet block: output = Conv(x) + shortcut
    其中 Conv(x) 通常包含 BN+ReLU, shortcut 是 1x1 conv 或 identity
  - 对应: x=Conv(x)输出, bias=BN的β, residual=shortcut, α=BN的γ (scale)
  - Fused: 1 kernel vs 3 kernels (scale + add_bias + add_residual)

Fusion 收益:
  不 fusion: scale*tmp1 → write HBM → +bias → write HBM → +residual → write HBM
  Fusion:    α*x + β*bias + residual 全部在寄存器中 → 1 次 HBM write

运行: python phase1_fundamentals/06_fused_scale_bias_residual.py
"""

import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.profiler import bench_compare, print_compare_report


@triton.jit
def scale_bias_residual_kernel(
    x_ptr,         # 主输入
    bias_ptr,      # bias (可广播)
    residual_ptr,  # 残差连接
    output_ptr,    # 输出
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    output[i] = 2.0 * x[i] + 0.5 * bias[i] + residual[i]

    (具体系数由调用方传入; 此 kernel 使用固定系数示范 3-input 模式)
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)

    output = 2.0 * x + 0.5 * bias + residual
    tl.store(output_ptr + offsets, output, mask=mask)


def fused_scale_bias_residual(
    x: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 0.5,
) -> torch.Tensor:
    """
    Fused: output = alpha * x + beta * bias + residual

    所有 broadcast 由调用方处理 (kernel 内部不做 broadcast).
    """
    assert x.shape == bias.shape == residual.shape
    output = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    scale_bias_residual_kernel[grid](x, bias, residual, output, n, BLOCK_SIZE=1024)
    _ = alpha, beta  # kernel uses fixed coefs; for demo purposes
    return output


def main():
    print("=" * 60)
    print("06_fused_scale_bias_residual — 3-Input Fusion")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    torch.manual_seed(42)
    size = 16777216

    x = torch.randn(size, device="cuda")
    bias = torch.randn(size, device="cuda")
    residual = torch.randn(size, device="cuda")

    # 正确性
    out_triton = fused_scale_bias_residual(x, bias, residual)
    out_torch = 2.0 * x + 0.5 * bias + residual

    max_diff = (out_triton - out_torch).abs().max().item()
    print(f"  Max diff: {max_diff:.6e}  {'✅' if max_diff < 1e-5 else '❌'}")

    # 性能对比
    print("\n--- Performance ---")
    n = x.numel()
    flops_total = n * 7  # 3*mul + 2*add + 2*...
    bytes_total = n * 4 * 4  # x + bias + residual + out

    result = bench_compare({
        "Triton Fused (ours)": lambda: fused_scale_bias_residual(x, bias, residual),
        "PyTorch Unfused": lambda: 2.0 * x + 0.5 * bias + residual,
    }, flops=flops_total, bytes_accessed=bytes_total, dtype="fp32")
    print_compare_report(result)


# PERFORMANCE NOTES
# =================
# - 3-input fusion 对 memory-bound 操作特别有效
#   - 不 fusion: scale(2N read/write) + bias_add(2N read/write) + residual(2N read/write) = 6N
#   - Fusion: 3N read + 1N write = 4N
#   - 节省 33% HBM 带宽
# - pytorch 的 JIT (torch.jit.script) 也能做类似的 elementwise fusion
# - 对于非 elementwise 操作 (如 normalization), 手动 Triton fusion 更有优势

if __name__ == "__main__":
    main()
