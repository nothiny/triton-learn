"""
35_relu6_clamp.py — ReLU6 Activation (Saturating Clamp)

学习目标:
  - 掌握 min/max clamp 的 GPU 实现: 两次无分支比较
  - 理解 ReLU6 = min(max(0, x), 6.0) 为什么有利于 int8 量化
  - 学习 clamp 模式: 截断值域, 减少表示范围换取更高精度

为什么需要 ReLU6:
  - MobileNetV1/V2 使用 ReLU6 替代 ReLU
  - ReLU6 把输出限制在 [0, 6], 在 int8 量化时:
    ReLU 无上界 → 需要大范围 → 量化步长粗 → 精度低
    ReLU6 上界=6 → 范围小 → 量化步长细 → 精度高
  - 6.0 的经验选择: 足够大, 不会截断太多信息

和 04_leaky_relu 的关系:
  - LeakyReLU: max(alpha*x, x) — 单次 compare
  - ReLU6: min(max(0, x), 6.0) — 两次 compare (sandwich)

运行: python phase1_fundamentals/35_relu6_clamp.py
"""

import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch, triton, triton.language as tl
from utils.profiler import bench_compare, print_compare_report


@triton.jit
def relu6_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """output = min(max(0, x), 6.0) — 饱和 clamp"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # [GPU] 两次 PTX max/min 指令: max.f32, min.f32 — 都是无分支单周期
    # ReLU: max(0, x)
    # + 上限: min(result, 6.0)
    clamped = tl.minimum(tl.maximum(x, 0.0), 6.0)

    tl.store(output_ptr + offsets, clamped, mask=mask)


def relu6(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    relu6_kernel[grid](x, output, n, BLOCK_SIZE=1024)
    return output


def main():
    print("=" * 60)
    print("35_relu6_clamp — ReLU6 Activation (min(max(0,x), 6.0))")
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.manual_seed(42)

    for name, size in [("small ", 256), ("medium", 65536), ("large ", 1048576)]:
        x = torch.randn(size, device="cuda") * 10  # 范围 [-30, 30]
        y_t = relu6(x)
        y_r = torch.clamp(x, 0.0, 6.0)
        max_diff = (y_t - y_r).abs().max().item()
        print(f"  [{name}] size={size:8d}  max_diff={max_diff:.2e}  "
              f"{'✅' if max_diff < 1e-6 else '❌'}")

    # Visual check: show value distribution
    x = torch.linspace(-10, 10, 21, device="cuda")
    y_t = relu6(x)
    print(f"\n  Input range: [{x[0].item():.1f}, {x[-1].item():.1f}]")
    print(f"  Output: {y_t.tolist()}")
    print(f"  Expected: all in [0, 6], outside->clamped")

    print("\n--- Performance ---")
    x = torch.randn(16777216, device="cuda", dtype=torch.float32) * 10
    n = x.numel()
    result = bench_compare(
        {
            "Triton ReLU6": lambda: relu6(x),
            "PyTorch clamp(0,6)": lambda: torch.clamp(x, 0.0, 6.0),
            "PyTorch F.relu6": lambda: torch.nn.functional.relu6(x),
        },
        flops=n * 2,        # max + min
        bytes_accessed=n * 8,  # read + write
        dtype="fp32",
    )
    print_compare_report(result)


# PERFORMANCE NOTES
# =================
# - ReLU6 = min(max(0, x), 6.0), 两次无分支 PTX max/min 指令.
# - Memory-bound: 2 ops / 8 bytes = 0.25 FLOP/byte, 和 identity copy 速度几乎一样.
# - ReLU6 对比 ReLU: 多了 1 次 min 指令, 时间几乎无增加 (都不是瓶颈).
# - 应用: MobileNet, int8 量化模型的关键激活函数.
# - 和后续 36_hard_sigmoid, 37_hard_swish 属于 MobileNet 激活三件套.

if __name__ == "__main__":
    main()
