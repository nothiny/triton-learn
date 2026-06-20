"""
37_hard_swish.py — Hard Swish Activation (MobileNetV3)

学习目标:
  - 掌握复合激活函数的 GPU 实现: x * hard_sigmoid(x)
  - 理解 "复合计算在单 kernel 内完成" 的融合优势
  - 学习 MobileNetV3 的核心激活, 为高效推理打基础

数学定义:
  swish(x) = x * sigmoid(x)                          (精确)
  hard_swish(x) = x * clamp((x+3)/6, 0, 1)           (近似)

MobileNetV3 为什么用 Hard Swish:
  1. sigmoid 的 exp 太贵 (推理瓶颈)
  2. Hard Swish = SiLU 的 cheap 替代
  3. 只在网络后半段用 (前半段用 ReLU6), 平衡精度和速度

和 07_silu 的关系:
  - SiLU/Swish = x * sigmoid(x)       (07 已实现)
  - Hard Swish = x * hard_sigmoid(x)  (本文件)
  - SwiGLU = gate * SiLU(up)         (10 已实现)

运行: python phase1_fundamentals/37_hard_swish.py
"""

import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch, triton, triton.language as tl
from utils.profiler import bench_compare, print_compare_report


@triton.jit
def hard_swish_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """output = x * clamp(x/6 + 0.5, 0, 1) = x * hard_sigmoid(x)"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # [GPU] 全部在寄存器中完成: 1 load, 5 ops, 1 store
    # FMA(x * 1/6 + 0.5) → clamp → mul by x
    hard_sig = tl.minimum(tl.maximum(x * (1.0 / 6.0) + 0.5, 0.0), 1.0)
    result = x * hard_sig

    tl.store(output_ptr + offsets, result, mask=mask)


def hard_swish(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    hard_swish_kernel[grid](x, output, n, BLOCK_SIZE=1024)
    return output


def main():
    print("=" * 60)
    print("37_hard_swish — Hard Swish Activation (MobileNetV3)")
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.manual_seed(42)

    for name, size in [("small ", 256), ("medium", 65536), ("large ", 1048576)]:
        x = torch.randn(size, device="cuda") * 6
        y_t = hard_swish(x)
        y_r = x * torch.clamp((x + 3) / 6, 0, 1)
        max_diff = (y_t - y_r).abs().max().item()
        print(f"  [{name}] size={size:8d}  max_diff={max_diff:.2e}  "
              f"{'✅' if max_diff < 1e-6 else '❌'}")

    # Compare Hard Swish vs SiLU (exact swish) vs ReLU6
    print("\n--- Activation Comparison ---")
    x = torch.linspace(-4, 4, 9, device="cuda")
    y_hswish = hard_swish(x)
    y_silu = torch.nn.functional.silu(x)
    y_relu6 = torch.clamp(x, 0, 6)
    print(f"  {'x':>8s}  {'HardSwish':>10s}  {'SiLU':>10s}  {'ReLU6':>10s}")
    for i in range(len(x)):
        print(f"  {x[i].item():8.2f}  {y_hswish[i].item():10.4f}  "
              f"{y_silu[i].item():10.4f}  {y_relu6[i].item():10.4f}")

    print("\n--- Performance ---")
    x = torch.randn(16777216, device="cuda", dtype=torch.float32) * 6
    n = x.numel()
    result = bench_compare(
        {
            "Triton Hard Swish": lambda: hard_swish(x),
            "PyTorch hard_swish": lambda: torch.nn.functional.hardswish(x),
            "PyTorch F.silu (exact)": lambda: torch.nn.functional.silu(x),
        },
        flops=n * 5,          # mul + mul + add + min + max
        bytes_accessed=n * 8,  # read + write
        dtype="fp32",
    )
    print_compare_report(result)


# PERFORMANCE NOTES
# =================
# - Hard Swish = x * clamp(x/6+0.5, 0, 1), 5 ops in registers.
# - 比 SiLU (x*sigmoid(x)) 快约 2-3x (省了 MUFU exp).
# - Memory-bound: 所有计算在寄存器内, 瓶颈是 HBM 带宽.
# - MobileNetV3 只在后半段 (>1/3 layers) 用 Hard Swish,
#   因为 Hard Swish 在低分辨率时有微小的精度损失.
# - 和 36_hard_sigmoid 的关系: Hard Swish = x * Hard Sigmoid(x).

if __name__ == "__main__":
    main()
