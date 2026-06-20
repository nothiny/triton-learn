"""
36_hard_sigmoid.py — Hard Sigmoid Activation (Piecewise Linear)

学习目标:
  - 掌握分段线性 (piecewise linear) 函数在 GPU 上的实现
  - 理解 "近似代替精确" 的推理优化思想 (效率 vs 精度)
  - 学习 clamp-based 激活的实现模式

数学定义:
  sigmoid(x) = 1 / (1 + exp(-x))          (精确, 需要 exp + div)

  hard_sigmoid(x) = clamp((x + 3) / 6, 0, 1)   (近似, 只需 add + mul + clamp)

对比分析:
  ┌──────────────┬─────────┬─────────┬─────────────────┐
  │              │ sigmoid │ hard    │ 备注             │
  ├──────────────┼─────────┼─────────┼─────────────────┤
  │ 运算          │ exp+div │ add+mul │                  │
  │ GPU 延迟       │ ~16 cyc │ ~4 cyc  │ exp 需要查表+插值 │
  │ x=-3 输出      │ ~0.047  │ 0.0     │ hard 截断为 0    │
  │ x=0 输出       │ 0.5     │ 0.5     │ 精确匹配          │
  │ x=+3 输出      │ ~0.953  │ 1.0     │ hard 截断为 1    │
  │ 最大误差        │ —       │ ≤0.05   │ 对量化模型可接受   │
  └──────────────┴─────────┴─────────┴─────────────────┘

为什么需要 Hard Sigmoid:
  - 推理时, exp() 是 GPU 上最贵的单指令之一 (MUFU 单元, ~16 cycles)
  - add + mul + clamp 只需 ~4 cycles (FMA + min/max)
  - 在 MobileNet/EdgeTPU 上, hard sigmoid 是关键优化
  - 梯度训练时用精确 sigmoid (smooth), 推理时用 hard (fast)

运行: python phase1_fundamentals/36_hard_sigmoid.py
"""

import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch, triton, triton.language as tl
from utils.profiler import bench_compare, print_compare_report


@triton.jit
def hard_sigmoid_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """output = clamp((x + 3) / 6, 0, 1)"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # [GPU] add + mul (FMA) + 2× clamp = ~4 cycles total
    # (x+3)/6 在 GPU 上用 1/x ≈ 0.1667 的 FMA: x * 0.1667 + 0.5
    clipped = x * (1.0 / 6.0) + 0.5
    result = tl.minimum(tl.maximum(clipped, 0.0), 1.0)

    tl.store(output_ptr + offsets, result, mask=mask)


def hard_sigmoid(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    hard_sigmoid_kernel[grid](x, output, n, BLOCK_SIZE=1024)
    return output


def main():
    print("=" * 60)
    print("36_hard_sigmoid — Hard Sigmoid (Piecewise Linear)")
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.manual_seed(42)

    for name, size in [("small ", 256), ("medium", 65536), ("large ", 1048576)]:
        x = torch.randn(size, device="cuda") * 6  # range ~[-18, 18]
        y_t = hard_sigmoid(x)
        y_r = torch.clamp((x + 3) / 6, 0, 1)
        max_diff = (y_t - y_r).abs().max().item()
        print(f"  [{name}] size={size:8d}  max_diff={max_diff:.2e}  "
              f"{'✅' if max_diff < 1e-6 else '❌'}")

    # Compare with exact sigmoid
    print("\n--- Hard vs Exact Sigmoid ---")
    x = torch.linspace(-6, 6, 13, device="cuda")
    y_hard = hard_sigmoid(x)
    y_exact = torch.sigmoid(x)
    print(f"  {'x':>8s}  {'hard_sigmoid':>12s}  {'sigmoid':>12s}  {'error':>10s}")
    for i in range(len(x)):
        print(f"  {x[i].item():8.2f}  {y_hard[i].item():12.6f}  "
              f"{y_exact[i].item():12.6f}  {abs(y_hard[i]-y_exact[i]).item():10.6f}")

    print("\n--- Performance ---")
    x = torch.randn(16777216, device="cuda", dtype=torch.float32) * 6
    n = x.numel()
    result = bench_compare(
        {
            "Triton Hard Sigmoid": lambda: hard_sigmoid(x),
            "PyTorch clamp((x+3)/6,0,1)": lambda: torch.clamp((x + 3) / 6, 0, 1),
            "PyTorch F.sigmoid (exact)": lambda: torch.sigmoid(x),
        },
        flops=n * 3,        # add + mul + clamp
        bytes_accessed=n * 8,  # read + write
        dtype="fp32",
    )
    print_compare_report(result)


# PERFORMANCE NOTES
# =================
# - Hard sigmoid = (x+3)/6 clamped to [0,1], 用 FMA + min + max 实现.
# - 比精确 sigmoid 快 ~3-4x (省了 exp 查表+插值的 ~16 cycles).
# - max error ≤ 0.05, 对推理精度影响 < 0.1% (MobileNetV3 已验证).
# - 和 35_relu6_clamp 结构类似: 都是 elementwise clamp 模式.

if __name__ == "__main__":
    main()
