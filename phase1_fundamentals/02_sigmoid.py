"""
02_sigmoid.py — Sigmoid activation kernel

学习目标：
  - 第一个激活函数 kernel: 理解 elementwise 操作的基础模式
  - 掌握 tl.sigmoid 的使用（利用 GPU MUFU 硬件）
  - 对比手写 exp 版 sigmoid vs 硬件加速版

数学公式:
  sigmoid(x) = 1 / (1 + exp(-x))

数值稳定性:
  - x >> 0: exp(-x) ≈ 0, sigmoid(x) ≈ 1
  - x << 0: exp(-x) → ∞, sigmoid(x) ≈ 0
  - tl.sigmoid 内部已处理溢出, 使用硬件指令 MUFU

运行: python phase1_fundamentals/02_sigmoid.py
"""

import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.profiler import bench_compare, print_compare_report


@triton.jit
def sigmoid_kernel(
    x_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Sigmoid: output[i] = 1 / (1 + exp(-x[i]))"""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # tl.sigmoid 利用 GPU MUFU 硬件, 比手写 exp 更快更准
    output = tl.sigmoid(x)
    tl.store(output_ptr + offsets, output, mask=mask)


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    sigmoid_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output


def main():
    print("=" * 60)
    print("02_sigmoid — Sigmoid Activation")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    torch.manual_seed(42)

    for name, x in [
        ("normal", torch.randn(1024, 4096, device="cuda")),
        ("large-pos", torch.full((256, 128), 20.0, device="cuda")),
        ("large-neg", torch.full((256, 128), -20.0, device="cuda")),
    ]:
        out_triton = sigmoid(x)
        out_torch = torch.sigmoid(x)
        max_diff = (out_triton - out_torch).abs().max().item()
        print(f"  [{name:>12s}] max_diff={max_diff:.2e}  "
              f"{'✅' if max_diff < 1e-5 else '❌'}")

    print("\n--- Performance ---")
    x = torch.randn(16777216, device="cuda", dtype=torch.float32)
    n = x.numel()
    result = bench_compare({
        "Triton (ours)": lambda: sigmoid(x),
        "PyTorch (ref)": lambda: torch.sigmoid(x),
    }, flops=n * 4, bytes_accessed=n * 2 * 4, dtype="fp32")
    print_compare_report(result)


# PERFORMANCE NOTES
# =================
# - sigmoid 是 memory-bound: ~4 FLOPs / 8 bytes = 0.5 FLOP/byte
# - tl.sigmoid 编译为 MUFU (Multi-Function Unit) 硬件指令
# - MUFU 在 SM 内部专门处理 exp/log/rsqrt 等特殊函数, 不占用 FP32/FMA 单元

if __name__ == "__main__":
    main()
