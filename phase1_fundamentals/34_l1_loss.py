"""
34_l1_loss.py — Fused L1 Loss (MAE) Kernel

学习目标:
  - 掌握 L1 Loss = sum(|pred - target|) / N 的融合实现
  - 理解 tl.abs 在 GPU 上的实现 (符号位清除, 非分支)
  - 对比 L1 vs MSE: L1 更鲁棒 (对 outlier 不敏感)

数学定义:
  L1_loss(pred, target) = (1/N) * sum(|pred[i] - target[i]|)

GPU 实现:
  tl.abs(x) → PTX abs.f32 (单周期, 清除符号位)
  不是 if x < 0: x = -x, 而是 x & 0x7FFFFFFF (位操作)

L1 vs MSE vs Hinge:
  - L1: 所有错误都惩罚, 线性增长
  - MSE: 大错误惩罚更重 (平方), 对小错误宽容
  - Hinge: 只惩罚 margin 内的错误, margin 外零惩罚

运行: python phase1_fundamentals/34_l1_loss.py
"""

import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch, triton, triton.language as tl
from utils.profiler import bench_compare, print_compare_report


@triton.jit
def l1_loss_kernel(pred_ptr, target_ptr, output_ptr, n_elements,
                    BLOCK_SIZE: tl.constexpr):
    """output[0] = sum(|pred - target|)"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    p = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
    t = tl.load(target_ptr + offsets, mask=mask, other=0.0)

    # [GPU] tl.abs → PTX abs.f32, 清除符号位, 无分支
    partial = tl.sum(tl.abs(p - t), axis=0)
    tl.atomic_add(output_ptr, partial)


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Fused L1 loss: single-pass |diff| + reduction."""
    assert pred.shape == target.shape
    n = pred.numel()
    abs_sum = torch.zeros(1, device=pred.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    l1_loss_kernel[grid](pred, target, abs_sum, n, BLOCK_SIZE=1024)
    return abs_sum / n


def main():
    print("=" * 60)
    print("34_l1_loss — Fused L1 Loss (MAE)")
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.manual_seed(42)

    for name, size in [("small ", 256), ("medium", 65536), ("large ", 1048576)]:
        pred = torch.randn(size, device="cuda")
        target = torch.randn(size, device="cuda")
        l1_t = l1_loss(pred, target).item()
        l1_r = torch.nn.functional.l1_loss(pred, target).item()
        err = abs(l1_t - l1_r)
        print(f"  [{name}] size={size:8d}  L1={l1_t:.6f}/{l1_r:.6f}  "
              f"diff={err:.2e}  {'✅' if err < 1e-5 else '❌'}")

    # Robustness demo: L1 vs MSE on data with outlier
    print("\n--- Robustness: L1 vs MSE with outliers ---")
    x = torch.randn(1000, device="cuda")
    # Add a single outlier
    x[0] = 100.0
    y = torch.zeros(1000, device="cuda")
    l1 = l1_loss(x, y).item()
    mse = torch.nn.functional.mse_loss(x, y).item()
    print(f"  L1={l1:.4f}  MSE={mse:.4f}  (outlier doubles MSE, barely affects L1)")

    print("\n--- Performance ---")
    pred = torch.randn(16777216, device="cuda", dtype=torch.float32)
    target = torch.randn(16777216, device="cuda", dtype=torch.float32)
    n = pred.numel()
    result = bench_compare(
        {
            "Triton fused L1": lambda: l1_loss(pred, target),
            "PyTorch F.l1_loss": lambda: torch.nn.functional.l1_loss(pred, target),
        },
        flops=n * 2,          # sub + abs
        bytes_accessed=n * 8,  # read pred + target
        dtype="fp32",
    )
    print_compare_report(result)


# PERFORMANCE NOTES
# =================
# - tl.abs 编译为 PTX abs.f32 (符号位清除), 和 add/mul 一样是单周期指令.
# - L1 loss 比 MSE 少一次 mul (diff² → |diff|), 但 bandwidth 是瓶颈, 减少 1 FLOP 无明显差别.
# - 和 32_mse_loss 结构完全一样, 只有 elementwise op 不同.
#   → 展示了 "loss function kernel 的模板化" 思路.

if __name__ == "__main__":
    main()
