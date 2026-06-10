"""
06_silu.py — SiLU / Swish activation kernel

学习目标：
  - 理解 SiLU (Sigmoid Linear Unit) 的数学和实现
  - 掌握 Triton 中 elementwise + 复杂数学函数的写法
  - 了解 SiLU 在现代 LLM 架构中的角色（SwiGLU 的组成部分）

数学公式:
  SiLU(x) = x * σ(x) = x / (1 + exp(-x))
  其中 σ(x) 是 sigmoid 函数

在深度学习中的使用:
  - SiLU (也叫 Swish) 是 Llama, PaLM 等模型的激活函数
  - SwiGLU(x, gate) = SiLU(x) * gate (见 Liger 的融合实现)
  - 比 ReLU 更平滑 → 梯度流更好 → 训练更稳定
  - 缺点: 比 ReLU 计算量更大 (需要 exp)

数值稳定性:
  - x >> 0 时: sigmoid(x) ≈ 1, SiLU(x) ≈ x
  - x << 0 时: sigmoid(x) ≈ 0, SiLU(x) ≈ 0
  - Triton 的 tl.sigmoid 内部已处理 exp 溢出

运行: python phase1_fundamentals/07_silu.py
"""

import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.profiler import bench_compare, print_compare_report


@triton.jit
def silu_kernel(
    x_ptr,       # 输入指针
    output_ptr,  # 输出指针
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    SiLU 激活: output[i] = x[i] * sigmoid(x[i])

    纯 elementwise 操作，每个线程独立计算。
    没有跨线程通信 → 理想的并行计算。
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # 从 HBM 加载
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # SiLU: x * sigmoid(x)
    # tl.sigmoid 已在 Triton 中实现，内部处理数值稳定性
    output = x * tl.sigmoid(x)

    # 写回 HBM
    tl.store(output_ptr + offsets, output, mask=mask)


def silu(x: torch.Tensor) -> torch.Tensor:
    """
    SiLU 激活函数 (又名 Swish)。

    Args:
        x: 输入张量

    Returns:
        x * sigmoid(x)
    """
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    silu_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output


def main():
    print("=" * 60)
    print("06_silu — SiLU/Swish Activation")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # ---- 正确性测试 ----
    torch.manual_seed(42)

    # 测试各种范围的输入（包括极端值）
    test_inputs = [
        ("normal", torch.randn(1024, 4096, device="cuda")),
        ("large-positive", torch.full((1024, 256), 20.0, device="cuda")),
        ("large-negative", torch.full((1024, 256), -20.0, device="cuda")),
        ("zeros", torch.zeros(1024, 256, device="cuda")),
    ]

    for name, x in test_inputs:
        out_triton = silu(x)
        out_torch = torch.nn.functional.silu(x)

        max_diff = (out_triton - out_torch).abs().max().item()
        status = "✅" if max_diff < 1e-4 else "❌"
        print(f"  [{name:>16s}] shape={list(x.shape)}  max_diff={max_diff:.2e}  {status}")

    # ---- 性能对比: Triton vs PyTorch ----
    print("\n--- Performance ---")

    x = torch.randn(16777216, device="cuda", dtype=torch.float32)  # 16M

    n_elements = x.numel()
    # SiLU: sigmoid (~4 FLOPs) + mul (1 FLOP) ≈ 5 FLOPs per element
    flops_total = n_elements * 5
    # x(4B) read + out(4B) write
    bytes_total = n_elements * 2 * 4

    result = bench_compare(
        {
            "Triton (ours)": lambda: silu(x),
            "PyTorch (ref)": lambda: torch.nn.functional.silu(x),
        },
        flops=flops_total,
        bytes_accessed=bytes_total,
        dtype="fp32",
    )
    print_compare_report(result)


# PERFORMANCE NOTES
# =================
# - SiLU 是 memory-bound: 5 FLOPs / 8 bytes = 0.625 FLOP/byte
#   H100 ridge point ≈ 295 FLOP/byte → 明确 memory-bound
# - Elementwise kernel 极其简单: 无共享内存, 无 reduction, 无线程同步
# - Triton 和 PyTorch 的 elementwise kernel 性能几乎相同
#   因为瓶颈都在 HBM 带宽, 计算部分微不足道
# - 优化的方向是 operator fusion:
#   - 将 SiLU 和上游/下游操作融合 (如 SwiGLU 已在 Liger 中实现)
#   - 单 kernel 中做 matmul + SiLU → 减少 memory round-trip
# - tl.sigmoid 内部实现 (PTX):
#   - 利用 MUFU (Multi-Function Unit) 的 special function unit
#   - 比手写 exp 再手动除法更快
# - 在训练中使用 SiLU 替代 ReLU 的主要原因是:
#   - 平滑性: SiLU 在 x=0 附近可微 (ReLU 在 0 处不可微)
#   - 非单调性: 在负半轴有小的负值输出 → 更丰富的梯度信号

if __name__ == "__main__":
    main()
