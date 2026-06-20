"""
32_mse_loss.py — Fused MSE Loss Kernel

学习目标:
  - 掌握 loss function 的 GPU 实现模式: elementwise + reduction 融合
  - 理解 MSE = sum((pred - target)²) / N 如何在一次 GPU pass 中完成
  - 学习 "先 compute, 后 reduce" 的通用 fusion 模式

数学定义:
  MSE(pred, target) = (1/N) * sum((pred[i] - target[i])²)

为什么需要 fused MSE:
  - PyTorch nn.MSELoss: pred - target → pow2 → mean (3 kernel launches)
  - Triton fused: 一次读取, 在一个 kernel 内完成 diff+square+mean
  - 减少 HBM 读写, 对 bandwidth-bound 场景提升明显

和 reduction 系列的关系:
  - 本质是 12_vector_sum 的 "元素先做运算" 版本
  - diff² 部分 = elementwise, sum/N 部分 = reduction

运行: python phase1_fundamentals/32_mse_loss.py
"""

import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch, triton, triton.language as tl
from utils.profiler import bench_compare, print_compare_report


@triton.jit
def mse_loss_kernel(pred_ptr, target_ptr, output_ptr, n_elements,
                     BLOCK_SIZE: tl.constexpr):
    """output[0] = sum((pred - target)²) / N"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    p = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
    t = tl.load(target_ptr + offsets, mask=mask, other=0.0)

    diff = p - t
    partial = tl.sum(diff * diff, axis=0)
    tl.atomic_add(output_ptr, partial)


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Fused MSE loss: single-pass compute + reduction."""
    assert pred.shape == target.shape
    n = pred.numel()
    sum_sq = torch.zeros(1, device=pred.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    mse_loss_kernel[grid](pred, target, sum_sq, n, BLOCK_SIZE=1024)
    return sum_sq / n


def main():
    print("=" * 60)
    print("32_mse_loss — Fused MSE Loss")
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.manual_seed(42)

    for name, size in [("small ", 256), ("medium", 65536), ("large ", 1048576)]:
        pred = torch.randn(size, device="cuda")
        target = torch.randn(size, device="cuda")
        mse_t = mse_loss(pred, target).item()
        mse_r = torch.nn.functional.mse_loss(pred, target).item()
        err = abs(mse_t - mse_r)
        print(f"  [{name}] size={size:8d}  mse={mse_t:.6f}/{mse_r:.6f}  "
              f"diff={err:.2e}  {'✅' if err < 1e-5 else '❌'}")

    print("\n--- Performance vs PyTorch ---")
    pred = torch.randn(16777216, device="cuda", dtype=torch.float32)
    target = torch.randn(16777216, device="cuda", dtype=torch.float32)
    n = pred.numel()
    result = bench_compare(
        {
            "Triton fused MSE": lambda: mse_loss(pred, target),
            "PyTorch nn.MSELoss": lambda: torch.nn.functional.mse_loss(pred, target),
        },
        flops=n * 3,          # sub + mul + add
        bytes_accessed=n * 8,  # read pred + target
        dtype="fp32",
    )
    print_compare_report(result)


# PERFORMANCE NOTES
# =================
# - Fused MSE: 一次 kernel 完成 diff + square + sum, 替代 PyTorch 的多次 launch.
# - Memory-bound: 读 2× 写 1× (atomic), arithmetic intensity = 3/8 = 0.375 FLOP/byte.
# - 和 12_vector_sum 类似, 但多了 elementwise compute (diff²), 这几乎不增加时间.
# - 实际训练中, MSE loss 通常和 backward 一起考虑; 这里只展示 forward.

if __name__ == "__main__":
    main()
