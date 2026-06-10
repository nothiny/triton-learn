"""
04_leaky_relu.py — Leaky ReLU / PReLU activation kernel

学习目标：
  - 理解 Leaky ReLU 和 PReLU 的差异
  - 掌握带参数的 elementwise kernel 写法
  - 学习 max/min/clamp 类操作的 Triton 实现

数学公式:
  LeakyReLU(x, α) = max(x, α*x)  where α 是固定负斜率 (如 0.01)
  PReLU(x, α)     = max(x, α*x)  where α 是可学习参数 (per-channel)

  等价写法: x if x > 0 else α*x

使用场景:
  - LeakyReLU: GAN, YOLO, 语音模型 (避免 dying ReLU)
  - PReLU: 让模型自己学 α (更多参数, 但更灵活)
  - "Dying ReLU": 当 ReLU 输入恒负时, 梯度恒 0, 神经元"死亡"
    LeakyReLU 在负半轴有小梯度 → 不会完全死亡

运行: python phase1_fundamentals/04_leaky_relu.py
"""

import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.profiler import bench_compare, print_compare_report


@triton.jit
def leaky_relu_kernel(
    x_ptr, output_ptr, n_elements,
    alpha: tl.constexpr,  # 负斜率
    BLOCK_SIZE: tl.constexpr,
):
    """
    LeakyReLU: output[i] = max(x[i], alpha * x[i])
              = x[i] if x[i] > 0 else alpha * x[i]

    alpha 是 constexpr → 编译器会将其编译为立即数。
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # tl.where 编译为 select 指令 (无分支 → 无 warp divergence)
    output = tl.where(x > 0, x, alpha * x)

    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def prelu_kernel(
    x_ptr, weight_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    PReLU (Parametric ReLU): output[i] = max(x[i], weight[c] * x[i])

    weight 是 per-channel 可学习参数. 这里简化为 1D weight (沿最后一维广播).
    对于 2D 输入 (N, C): weight[c] 作用于第 c 列.
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # PReLU weight: 对 (N, C) 输入, weight shape = (C,)
    # 用 modulo 计算每个元素的 channel 索引
    # 简化: 假设 1D 输入, weight 广播
    w = tl.load(weight_ptr + offsets, mask=mask, other=0.0)

    output = tl.where(x > 0, x, w * x)
    tl.store(output_ptr + offsets, output, mask=mask)


def leaky_relu(x: torch.Tensor, alpha: float = 0.01) -> torch.Tensor:
    """LeakyReLU: max(x, alpha*x)"""
    output = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    leaky_relu_kernel[grid](x, output, n, alpha=alpha, BLOCK_SIZE=1024)
    return output


def prelu(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """PReLU: max(x, weight*x), weight 可广播."""
    assert weight.numel() == x.shape[-1] or weight.numel() == 1
    output = torch.empty_like(x)
    n = x.numel()
    w = weight.expand_as(x).reshape(-1)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    prelu_kernel[grid](x.reshape(-1), w, output.reshape(-1), n, BLOCK_SIZE=1024)
    return output


def main():
    print("=" * 60)
    print("04_leaky_relu — LeakyReLU / PReLU Activation")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    torch.manual_seed(42)

    # ---- LeakyReLU 正确性 ----
    for alpha in [0.01, 0.1, 0.2]:
        x = torch.randn(1024, 4096, device="cuda")
        out_triton = leaky_relu(x, alpha)
        out_torch = torch.nn.functional.leaky_relu(x, alpha)
        max_diff = (out_triton - out_torch).abs().max().item()
        print(f"  [LeakyReLU α={alpha:.2f}] max_diff={max_diff:.2e}  "
              f"{'✅' if max_diff < 1e-5 else '❌'}")

    # ---- PReLU 正确性 ----
    x = torch.randn(512, 256, device="cuda")
    w = torch.rand(256, device="cuda") * 0.5
    out_triton = prelu(x, w)
    out_torch = torch.nn.functional.prelu(x, w)
    max_diff = (out_triton - out_torch).abs().max().item()
    print(f"  [PReLU (512,256)]      max_diff={max_diff:.2e}  "
          f"{'✅' if max_diff < 1e-3 else '❌'}")

    # ---- 性能对比 ----
    print("\n--- Performance (LeakyReLU, α=0.01) ---")
    x = torch.randn(16777216, device="cuda", dtype=torch.float32)
    n = x.numel()
    result = bench_compare({
        "Triton (ours)": lambda: leaky_relu(x, 0.01),
        "PyTorch (ref)": lambda: torch.nn.functional.leaky_relu(x, 0.01),
    }, flops=n * 3, bytes_accessed=n * 2 * 4, dtype="fp32")
    print_compare_report(result)


# PERFORMANCE NOTES
# =================
# - LeakyReLU 是 memory-bound: 3 FLOPs / 8 bytes
# - tl.where(cond, a, b) → LLVM select 指令 → 无分支, 无 warp divergence
# - LeakyReLU vs ReLU: 计算量几乎相同, 但 LeakyReLU 多一次乘法
# - PReLU 的 weight 是 n_channel 个参数, 几乎不增加计算开销
# - 为什么不用 LeakyReLU 替代 ReLU?
#   - 在深层网络中, dying ReLU 不是主要问题 (BatchNorm/LayerNorm 解决了)
#   - 现代架构用 SiLU/GELU → 平滑且非单调, 效果更好

if __name__ == "__main__":
    main()
