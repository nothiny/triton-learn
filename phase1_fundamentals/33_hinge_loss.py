"""
33_hinge_loss.py — Fused Hinge Loss Kernel

学习目标:
  - 掌握带比较操作的 loss function 模式: max(0, margin - y*pred)
  - 理解 tl.where / tl.maximum 在 GPU 上的零开销执行 (无分支)
  - 学习 SVM-type losses 的 GPU 实现

数学定义:
  hinge_loss(pred, target, margin) = sum(max(0, margin - y_i * pred_i)) / N
  其中 y_i ∈ {-1, +1} (二元标签)

为什么 GPU 上无分支:
  - tl.maximum(0, x) 编译为 PTX max 指令, 无分支
  - 等价于 max(0, margin - y*pred), 始终是 2 FLOPs/element

对比 MSE:
  - MSE 是连续的 quadratic loss, 对 outlier 敏感
  - Hinge 是 margin-based, 只惩罚 "不够确信" 的预测

运行: python phase1_fundamentals/33_hinge_loss.py
"""

import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch, triton, triton.language as tl
from utils.profiler import bench_compare, print_compare_report


@triton.jit
def hinge_loss_kernel(pred_ptr, target_ptr, output_ptr, n_elements, margin,
                       BLOCK_SIZE: tl.constexpr):
    """
    output[0] = sum(max(0, margin - target * pred)) / N
    target[i] ∈ {-1, +1}
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    pred = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0)

    # [GPU] tl.maximum → PTX max.f32, 无分支
    loss = tl.maximum(0.0, margin - target * pred)
    # [GPU] 必须 mask 掉越界元素, 否则 other=0.0 的 target 会贡献非零 loss
    loss = tl.where(mask, loss, 0.0)
    partial = tl.sum(loss, axis=0)
    tl.atomic_add(output_ptr, partial)


def hinge_loss(pred: torch.Tensor, target: torch.Tensor, margin: float = 1.0
               ) -> torch.Tensor:
    """Fused hinge loss: sum(max(0, margin - y*pred)) / N."""
    assert pred.shape == target.shape
    n = pred.numel()
    loss_sum = torch.zeros(1, device=pred.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    hinge_loss_kernel[grid](pred, target, loss_sum, n, margin, BLOCK_SIZE=1024)
    return loss_sum / n


def main():
    print("=" * 60)
    print("33_hinge_loss — Fused Hinge Loss")
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.manual_seed(42)

    for name, size in [("small ", 256), ("medium", 65536), ("large ", 1048576)]:
        pred = torch.randn(size, device="cuda")
        target = torch.where(torch.randn(size, device="cuda") > 0, 1.0, -1.0)
        hl_t = hinge_loss(pred, target).item()
        # Manual reference: hinge loss = max(0, 1 - y * pred).mean()
        hl_r = torch.clamp(1.0 - target * pred, min=0).mean().item()
        err = abs(hl_t - hl_r)
        print(f"  [{name}] size={size:8d}  hinge={hl_t:.6f}/{hl_r:.6f}  "
              f"diff={err:.2e}  {'✅' if err < 1e-5 else '❌'}")

    print("\n--- Performance ---")
    pred = torch.randn(16777216, device="cuda", dtype=torch.float32)
    target = torch.where(torch.randn(16777216, device="cuda") > 0, 1.0, -1.0)
    n = pred.numel()
    result = bench_compare(
        {
            "Triton fused Hinge": lambda: hinge_loss(pred, target),
            "PyTorch clamp(1-y*pred,0)": lambda: torch.clamp(
                1.0 - target * pred, min=0).mean(),
        },
        flops=n * 3,          # mul + sub + max
        bytes_accessed=n * 8,  # read pred + target
        dtype="fp32",
    )
    print_compare_report(result)


# PERFORMANCE NOTES
# =================
# - tl.maximum(0, x) 编译为 PTX 的 max.f32, 是单周期硬件指令, 无分支.
# - Hinge loss 和 MSE 结构几乎一样, 只是 elementwise 运算不同:
#   MSE: (pred-target)², Hinge: max(0, margin - y*pred).
# - GPU 上 "comparison" 和 "arithmetic" 指令的延迟相同 (都 4-8 cycles).
# - 所以带分支的 loss 和纯算术 loss 在 GPU 上速度几乎一样.

if __name__ == "__main__":
    main()
