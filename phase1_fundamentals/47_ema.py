"""
47_ema.py — Exponential Moving Average (EMA)

学习目标:
  - 掌握 EMA 的 GPU 实现: running = beta * running + (1-beta) * new
  - 理解 stateful kernel 的实现模式 (read-modify-write)
  - 学习 BatchNorm 和 model EMA 的底层原理

数学定义:
  ema = β * ema + (1-β) * new_value

  其中:
  - β (momentum): 通常 0.9 ~ 0.9999
  - ema: 历史平均值 (running mean/var)
  - new_value: 当前 batch 的统计量

应用场景:
  1. BatchNorm running_mean/running_var (β=0.9)
  2. Model EMA for evaluation (β=0.9999)
  3. Adam 的 first/second moment (β1=0.9, β2=0.999)
  4. Gradient moving average (momentum SGD)

为什么需要 EMA:
  - 直接用 batch mean 做 normalize → 训练/推理不一致
  - EMA 积累历史信息 → 推理时用 running_mean (更稳定)
  - β 越大 → 历史信息衰减越慢 → 越平滑但响应越慢

运行: python phase1_fundamentals/47_ema.py
"""

import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch, triton, triton.language as tl
from utils.profiler import bench_compare, print_compare_report


@triton.jit
def ema_update_kernel(running_ptr, new_ptr, n_elements, beta,
                        BLOCK_SIZE: tl.constexpr):
    """
    running = beta * running + (1 - beta) * new.
    In-place: 直接修改 running_ptr.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    old = tl.load(running_ptr + offsets, mask=mask)
    new = tl.load(new_ptr + offsets, mask=mask)

    # [GPU] FMA: beta*old + (1-beta)*new
    updated = beta * old + (1.0 - beta) * new

    tl.store(running_ptr + offsets, updated, mask=mask)


def ema_update(running: torch.Tensor, new: torch.Tensor, beta: float = 0.9
               ) -> None:
    """EMA: running = beta * running + (1-beta) * new. In-place on running."""
    assert running.shape == new.shape
    n = running.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    ema_update_kernel[grid](running, new, n, beta, BLOCK_SIZE=1024)


def main():
    print("=" * 60)
    print("47_ema — Exponential Moving Average")
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.manual_seed(42)

    # Correctness: compare with PyTorch
    beta = 0.9
    for name, size in [("small ", 256), ("medium", 65536), ("large ", 1048576)]:
        running = torch.randn(size, device="cuda")
        new = torch.randn(size, device="cuda")
        running_copy = running.clone()
        ema_update(running_copy, new, beta)
        expected = beta * running + (1 - beta) * new
        max_diff = (running_copy - expected).abs().max().item()
        print(f"  [{name}] size={size:8d}  max_diff={max_diff:.2e}  "
              f"{'✅' if max_diff < 1e-6 else '❌'}")

    # Simulation: show EMA smoothing over time
    print("\n--- EMA Simulation (beta=[0.5, 0.9, 0.99]) ---")
    for beta in [0.5, 0.9, 0.99]:
        x = torch.tensor([1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0, -3.0],
                         device="cuda")
        running = torch.zeros_like(x)
        for _ in range(5):  # 重复 5 次 (模拟 step)
            ema_update(running, x, beta)
        print(f"  β={beta:.2f} after 5× exposure: {running.tolist()}")

    print(f"\n  💡 β=0.99: 几乎不更新 (long memory)")
    print(f"     β=0.9:  适度更新 (BatchNorm default)")
    print(f"     β=0.5:  快速响应 (short memory)")

    print("\n--- Performance ---")
    running = torch.randn(16777216, device="cuda", dtype=torch.float32)
    new = torch.randn(16777216, device="cuda", dtype=torch.float32)
    n = running.numel()
    result = bench_compare(
        {
            "Triton EMA (in-place)": lambda: ema_update(running, new, beta),
            "PyTorch running = β*old+(1-β)*new": lambda: running.copy_(
                beta * running + (1 - beta) * new),
        },
        flops=n * 3,          # 2× mul + 1× add
        bytes_accessed=n * 4 * 3,  # read running + new, write running
        dtype="fp32",
    )
    print_compare_report(result)


# PERFORMANCE NOTES
# =================
# - EMA = FMA (beta*old) + (1-beta)*new = β·old + (1-β)·new.
# - In-place: 直接在 running buffer 上修改 (零额外分配).
# - Memory-bound: 3 ops / 12 bytes = 0.25 FLOP/byte.
# - 可以融合: 在 BatchNorm 训练中, EMA 和 mean/var 计算可以在同一 pass 中完成.
# - β 的选择: 训练前期用大 β (0.99), 后期用小 β (0.9) — 自适应 momentum.

if __name__ == "__main__":
    main()
