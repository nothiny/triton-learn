"""
03_tanh.py — Tanh activation kernel

学习目标：
  - 学会在 Triton 中手写数学函数（tanh 用 sigmoid 实现）
  - 理解 sigmoid/tanh 之间的数学等价关系
  - 掌握 GPU 上复合函数的写法

数学公式:
  tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
          = 2 * sigmoid(2x) - 1
          推导: sigmoid(y) = 1/(1+exp(-y))
                tanh(x) = (e^x-e^-x)/(e^x+e^-x) = (e^2x-1)/(e^2x+1)
                2*sigmoid(2x) = 2/(1+e^-2x) = 2e^2x/(e^2x+1)
                2*sigmoid(2x)-1 = (2e^2x-e^2x-1)/(e^2x+1) = (e^2x-1)/(e^2x+1) = tanh(x)

  为什么用 sigmoid 实现 tanh?
    - tl.sigmoid 有 MUFU 硬件加速
    - 避免手写 exp 的精度和性能问题

运行: python phase1_fundamentals/03_tanh.py
"""

import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.profiler import bench_compare, print_compare_report


@triton.jit
def tanh_kernel(
    x_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Tanh activation: output[i] = 2*sigmoid(2*x[i]) - 1

    [COMPILER] 2*sigmoid(2x)-1 编译为:
      MUFU.F2F.F32.SIGMOID r, 2*x
      FADD r, 2*r, -1.0
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    output = 2.0 * tl.sigmoid(2.0 * x) - 1.0
    tl.store(output_ptr + offsets, output, mask=mask)


def tanh(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    tanh_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output


def main():
    print("=" * 60)
    print("03_tanh — Tanh Activation (via sigmoid)")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    torch.manual_seed(42)

    for name, x in [
        ("normal", torch.randn(1024, 4096, device="cuda")),
        ("large-pos", torch.full((256, 128), 10.0, device="cuda")),
        ("large-neg", torch.full((256, 128), -10.0, device="cuda")),
        ("zero", torch.zeros(256, 128, device="cuda")),
    ]:
        out_triton = tanh(x)
        out_torch = torch.tanh(x)
        max_diff = (out_triton - out_torch).abs().max().item()
        print(f"  [{name:>12s}] max_diff={max_diff:.2e}  "
              f"{'✅' if max_diff < 1e-5 else '❌'}")

    print("\n--- Performance ---")
    x = torch.randn(16777216, device="cuda", dtype=torch.float32)
    n = x.numel()
    result = bench_compare({
        "Triton (ours)": lambda: tanh(x),
        "PyTorch (ref)": lambda: torch.tanh(x),
    }, flops=n * 5, bytes_accessed=n * 2 * 4, dtype="fp32")
    print_compare_report(result)


# PERFORMANCE NOTES
# =================
# - tanh 也是 memory-bound: 5 FLOPs / 8 bytes
# - 用 sigmoid 实现 tanh 比手写 exp 快: 利用 MUFU 硬件
# - tanh vs sigmoid:
#   - sigmoid 输出 [0, 1], tanh 输出 [-1, 1]
#   - tanh 在 RNN/LSTM 中常用 (门控机制)
#   - Transformer 中几乎不用 (都迁移到 GELU/SiLU 了)

if __name__ == "__main__":
    main()
