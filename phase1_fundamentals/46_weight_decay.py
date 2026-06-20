"""
46_weight_decay.py — In-Place Weight Decay (L2 Regularization)

学习目标:
  - 掌握 in-place 参数更新 (mutating input) 的 GPU 实现
  - 理解 weight decay 的数学本质: w = w * (1 - lr * wd)
  - 学习 "in-place" kernel 减少内存分配的模式

数学定义:
  weight_decay(w) = w * (1 - lr * weight_decay_rate)
  即 w -= lr * wd * w

为什么需要 in-place kernel:
  - PyTorch weight_decay: w = w * (1 - lr * wd) → 创建新 tensor, 再 copy
  - Triton in-place: 在旧 buffer 上直接修改, 零额外分配

为什么 Weight Decay ≠ L2 Regularization (在 AdamW 中):
  - L2 regularization: loss += wd * |w|², 通过梯度影响权重
  - Weight decay: w -= lr * wd * w, 直接在权重上衰减
  - 在 SGD 中等价, 但在 Adam/AdamW 中不等价 (因为 adaptive lr 改变了方向)

运行: python phase1_fundamentals/46_weight_decay.py
"""

import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch, triton, triton.language as tl
from utils.profiler import bench_compare, print_compare_report


@triton.jit
def weight_decay_kernel(w_ptr, n_elements, lr, wd,
                          BLOCK_SIZE: tl.constexpr):
    """
    In-place weight decay: w[i] = w[i] * (1 - lr * wd).
    替代 "w = w * (1 - lr*wd)" 的 Python 赋值.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    w = tl.load(w_ptr + offsets, mask=mask)
    # [GPU] FMA: w * (1 - lr*wd) = w - w * lr * wd
    decayed = w * (1.0 - lr * wd)
    tl.store(w_ptr + offsets, decayed, mask=mask)


def weight_decay_inplace(w: torch.Tensor, lr: float = 0.001,
                          wd: float = 0.01) -> None:
    """In-place weight decay. Modifies w directly, returns nothing."""
    n = w.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    weight_decay_kernel[grid](w, n, lr, wd, BLOCK_SIZE=1024)


def main():
    print("=" * 60)
    print("46_weight_decay — In-Place Weight Decay")
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.manual_seed(42)

    lr, wd = 0.001, 0.01
    decay_factor = 1.0 - lr * wd

    for name, size in [("small ", 256), ("medium", 65536), ("large ", 1048576)]:
        w = torch.randn(size, device="cuda")
        w_copy = w.clone()
        weight_decay_inplace(w, lr, wd)
        w_copy.mul_(decay_factor)
        max_diff = (w - w_copy).abs().max().item()
        print(f"  [{name}] size={size:8d}  max_diff={max_diff:.2e}  "
              f"{'✅' if max_diff < 1e-7 else '❌'}")

    # Demo: optimizer step with weight decay
    print("\n--- Optimizer Step with Weight Decay ---")
    w = torch.randn(5, device="cuda")
    print(f"  Before: {w.tolist()}")
    weight_decay_inplace(w, lr=0.01, wd=0.1)
    print(f"  After:  {w.tolist()}")

    print("\n--- Performance ---")
    w = torch.randn(16777216, device="cuda", dtype=torch.float32)
    n = w.numel()
    result = bench_compare(
        {
            "Triton weight_decay (in-place)": lambda: weight_decay_inplace(
                w, lr, wd),
            "PyTorch mul_(1-lr*wd)": lambda: w.mul_(decay_factor),
        },
        flops=n * 2,          # mul + sub
        bytes_accessed=n * 4 * 2,  # read + write (in-place)
        dtype="fp32",
    )
    print_compare_report(result)


# PERFORMANCE NOTES
# =================
# - In-place kernel: 读一次, 写一次 (同一 buffer), 零额外分配.
# - Weight decay 通常作为 optimizer step 的一部分, 可以和梯度更新融合.
# - 在 28_adamw 中, weight decay 是 AdamW 的第一步,
#   融合到同一个 kernel 中可以减少 HBM 读写.

if __name__ == "__main__":
    main()
