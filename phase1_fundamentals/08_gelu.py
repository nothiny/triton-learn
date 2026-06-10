"""
07_gelu.py — GELU (Gaussian Error Linear Unit) activation kernel

学习目标：
  - 理解 GELU 的数学推导（精确版 vs tanh 近似版）
  - 学习如何在 Triton 中实现复杂数学函数（tanh 近似）
  - 对比 SiLU vs GELU 的计算开销和特性

数学公式:
  精确版: GELU(x) = x * Φ(x)  where Φ 是标准正态分布的 CDF
           = x * 0.5 * (1 + erf(x / √2))

  tanh 近似 (PyTorch 默认):
    GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

在深度学习中的使用:
  - GELU 是 BERT, GPT-2/3, ViT 等 Transformer 模型的激活函数
  - 比 ReLU 平滑 → 更好的优化特性
  - 新模型倾向 SiLU: 数学更简单, 硬件友好, 效果相当

数值稳定性:
  - tanh 近似在 [-3, 3] 范围外和精确版有 ~1e-3 级别的误差
  - 对于推理/训练, 这个误差完全可以接受

运行: python phase1_fundamentals/08_gelu.py
"""

import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.profiler import bench_compare, print_compare_report


@triton.jit
def gelu_kernel(
    x_ptr,       # 输入指针
    output_ptr,  # 输出指针
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    GELU 激活 (tanh 近似):
        output[i] = 0.5 * x[i] * (1 + tanh(√(2/π) * (x[i] + 0.044715 * x[i]³)))

    每个线程独立计算，无跨线程通信。
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # 加载
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # ---- GELU tanh 近似 ----
    # Step 1: x³ (前向传播中允许负值，保留符号)
    x3 = x * x * x

    # Step 2: inner = √(2/π) * (x + 0.044715 * x³)
    # 常量: sqrt(2/π) ≈ 0.7978845608028654, 0.044715 是 tanh 近似的经验系数
    # [COMPILER] 局部 float 字面量自动被 Triton 视为编译时常量
    sqrt_2_over_pi = 0.7978845608028654
    gelu_coef = 0.044715
    inner = sqrt_2_over_pi * (x + gelu_coef * x3)

    # Step 3: 1 + tanh(inner)
    #        = 1 + (2*sigmoid(2*inner) - 1)
    #        = 2 * sigmoid(2 * inner)
    # [COMPILER] 使用 sigmoid 替代 tanh: Triton 有高效的 sigmoid 硬实现
    # Triton 的 tl.sigmoid 利用 GPU MUFU (Multi-Function Unit) 硬件
    tanh_val = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
    one_plus_tanh = 1.0 + tanh_val  # 等价于 2 * sigmoid(2*inner)

    # Step 4: 0.5 * x * (1 + tanh(...))
    output = 0.5 * x * one_plus_tanh

    tl.store(output_ptr + offsets, output, mask=mask)


def gelu(x: torch.Tensor) -> torch.Tensor:
    """
    GELU 激活函数 (tanh 近似)。

    Args:
        x: 输入张量

    Returns:
        GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    """
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    gelu_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output


def main():
    print("=" * 60)
    print("07_gelu — GELU Activation (tanh approx)")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # ---- 正确性测试 ----
    torch.manual_seed(42)

    # 各种输入范围
    test_inputs = [
        ("normal", torch.randn(1024, 4096, device="cuda")),
        ("uniform[-3,3]", torch.FloatTensor(1024, 256).uniform_(-3, 3).cuda()),
        ("large-pos", torch.full((256, 128), 10.0, device="cuda")),
        ("large-neg", torch.full((256, 128), -10.0, device="cuda")),
        ("zeros", torch.zeros(256, 128, device="cuda")),
    ]

    for name, x in test_inputs:
        out_triton = gelu(x)
        # PyTorch 默认也使用 tanh 近似
        out_torch = torch.nn.functional.gelu(x, approximate="tanh")

        max_diff = (out_triton - out_torch).abs().max().item()
        status = "✅" if max_diff < 1e-3 else "❌"
        print(f"  [{name:>16s}] shape={list(x.shape)}  max_diff={max_diff:.2e}  {status}")

    # ---- 性能对比: Triton vs PyTorch ----
    print("\n--- Performance ---")

    x = torch.randn(16777216, device="cuda", dtype=torch.float32)  # 16M

    n_elements = x.numel()
    # GELU: x³(2) + add(1) + mul(1) + sigmoid(~4) + mul(1) ≈ 9 FLOPs
    flops_total = n_elements * 9
    # x(4B) read + out(4B) write
    bytes_total = n_elements * 2 * 4

    result = bench_compare(
        {
            "Triton (ours)": lambda: gelu(x),
            "PyTorch (ref)": lambda: torch.nn.functional.gelu(x, approximate="tanh"),
        },
        flops=flops_total,
        bytes_accessed=bytes_total,
        dtype="fp32",
    )
    print_compare_report(result)


# PERFORMANCE NOTES
# =================
# - GELU 也是 memory-bound: ~9 FLOPs / 8 bytes ≈ 1.1 FLOP/byte
#   (H100 ridge point ≈ 295 FLOP/byte → 远在 memory-bound 区)
# - 计算开销对比:
#   - ReLU:     max(0, x)              ≈ 1 FLOP/elt  (最快)
#   - SiLU:     x * sigmoid(x)         ≈ 5 FLOP/elt
#   - GELU:     0.5*x*(1+tanh(inner))  ≈ 9 FLOP/elt  (最慢)
#   但实际运行时间三者几乎相同！因为瓶颈都在内存带宽
# - 为什么还有模型用 GELU？
#   - BERT/GPT 系列先于 SiLU 流行, 历史惯性
#   - 在特定任务上 GELU 可能略好 (有论文对比过)
# - 现代选择 (2024+):
#   - Llama/Mistral/Qwen → SiLU (via SwiGLU)
#   - 新的 Transformer 几乎都用 SiLU → 更简单更快
# - tanh 实现: 用 sigmoid 代替 tanh 避免了手写 exp
#   - tl.sigmoid 利用 MUFU 硬件, 比手写 exp 快
#   - 2*sigmoid(2x) - 1 是 tanh 的标准 sigmoid 等价形式

if __name__ == "__main__":
    main()
