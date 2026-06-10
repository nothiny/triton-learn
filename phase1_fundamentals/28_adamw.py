"""
28_adamw.py — AdamW Optimizer Step kernel

学习目标：
  - 理解 AdamW 优化器的数学公式
  - 掌握多 buffer elementwise update 的 kernel 设计
  - 体会 optimizer fusion 的实际加速价值

AdamW 更新公式 (对每个参数 p, 梯度 g):
  m = β₁ * m + (1-β₁) * g           ← 一阶动量 (first moment)
  v = β₂ * v + (1-β₂) * g²          ← 二阶动量 (second moment)
  m̂ = m / (1 - β₁ᵗ)                  ← 偏差校正 (bias correction)
  v̂ = v / (1 - β₂ᵗ)
  p -= lr * (m̂ / (√v̂ + ε) + wd * p)  ← 参数更新 (AdamW decoupled weight decay)

Adam vs AdamW:
  - Adam:  p -= lr * m̂ / (√v̂ + ε)          (L2 regularization 通过 grad)
  - AdamW: p -= lr * (m̂ / (√v̂ + ε) + wd*p)  (decoupled weight decay)
  - AdamW 的 weight decay 不经过 momentum → 更好的泛化

Fusion 收益:
  不 fusion: 6 个 CUDA kernel (m_update, v_update, bias_correct_m,
             bias_correct_v, scale_inv, p_update)
  Fusion:    1 个 CUDA kernel → 减少 5 次 HBM round-trip!
  实际加速: 2-3x vs PyTorch foreach (取决于参数规模)

运行: python phase1_fundamentals/28_adamw.py
"""

import math
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.profiler import bench_compare, print_compare_report


@triton.jit
def adamw_kernel(
    p_ptr,   # 参数 (in-place update)
    g_ptr,   # 梯度
    m_ptr,   # 一阶动量 (in-place update)
    v_ptr,   # 二阶动量 (in-place update)
    n_elements,
    lr: tl.constexpr,
    beta1: tl.constexpr,
    beta2: tl.constexpr,
    eps: tl.constexpr,
    weight_decay: tl.constexpr,
    bias_correction1: tl.constexpr,  # 1 - β₁ᵗ
    bias_correction2: tl.constexpr,  # 1 - β₂ᵗ
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused AdamW step: 单 kernel 完成 m, v, p 的全部更新.

    所有标量参数都是 constexpr → 编译器将它们嵌入为立即数.
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # 加载当前状态
    p = tl.load(p_ptr + offsets, mask=mask, other=0.0)
    g = tl.load(g_ptr + offsets, mask=mask, other=0.0)
    m_old = tl.load(m_ptr + offsets, mask=mask, other=0.0)
    v_old = tl.load(v_ptr + offsets, mask=mask, other=0.0)

    # ---- Step 1: 更新动量 ----
    m_new = beta1 * m_old + (1.0 - beta1) * g
    v_new = beta2 * v_old + (1.0 - beta2) * g * g

    # ---- Step 2: 偏差校正 ----
    m_hat = m_new / bias_correction1
    v_hat = v_new / bias_correction2

    # ---- Step 3: AdamW 参数更新 ----
    # p -= lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * p)
    update = m_hat / (tl.sqrt(v_hat) + eps) + weight_decay * p
    p_new = p - lr * update

    # 写回 (3 个 buffer 同时更新)
    tl.store(p_ptr + offsets, p_new, mask=mask)
    tl.store(m_ptr + offsets, m_new, mask=mask)
    tl.store(v_ptr + offsets, v_new, mask=mask)


def adamw_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,      # m
    exp_avg_sq: torch.Tensor,   # v
    step: int,
    lr: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01,
) -> None:
    """
    Fused AdamW update (in-place on param, exp_avg, exp_avg_sq).

    Args:
        param: 模型参数 (更新 in-place)
        grad: 梯度
        exp_avg: 一阶动量 m
        exp_avg_sq: 二阶动量 v
        step: 当前步数 (用于 bias correction)
    """
    assert param.shape == grad.shape == exp_avg.shape == exp_avg_sq.shape
    n = param.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    bias_correction1 = 1.0 - betas[0] ** step
    bias_correction2 = 1.0 - betas[1] ** step

    adamw_kernel[grid](
        param, grad, exp_avg, exp_avg_sq, n,
        lr=lr, beta1=betas[0], beta2=betas[1],
        eps=eps, weight_decay=weight_decay,
        bias_correction1=bias_correction1,
        bias_correction2=bias_correction2,
        BLOCK_SIZE=1024,
    )


def adamw_pytorch_step(param, grad, exp_avg, exp_avg_sq, step,
                       lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
    """PyTorch unfused reference implementation."""
    bias_correction1 = 1.0 - betas[0] ** step
    bias_correction2 = 1.0 - betas[1] ** step

    # 6 separate kernel launches in PyTorch's foreach implementation
    exp_avg.mul_(betas[0]).add_(grad, alpha=1 - betas[0])
    exp_avg_sq.mul_(betas[1]).addcmul_(grad, grad, value=1 - betas[1])

    denom = exp_avg_sq.sqrt().add_(eps)
    step_size = lr / bias_correction1
    param.addcdiv_(exp_avg, denom, value=-step_size)
    param.add_(param, alpha=-lr * weight_decay)


def main():
    print("=" * 60)
    print("28_adamw — Fused AdamW Optimizer Step")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    torch.manual_seed(42)

    # ---- 正确性 ----
    N = 4096
    p0 = torch.randn(N, device="cuda")
    g = torch.randn(N, device="cuda")
    m0 = torch.zeros(N, device="cuda")
    v0 = torch.zeros(N, device="cuda")

    # Triton AdamW
    p_t = p0.clone()
    m_t = m0.clone()
    v_t = v0.clone()
    adamw_step(p_t, g, m_t, v_t, step=10)

    # PyTorch reference
    p_r = p0.clone()
    m_r = m0.clone()
    v_r = v0.clone()
    adamw_pytorch_step(p_r, g, m_r, v_r, step=10)

    for name, a, b in [("p", p_t, p_r), ("m", m_t, m_r), ("v", v_t, v_r)]:
        max_diff = (a - b).abs().max().item()
        print(f"  {name}: max_diff={max_diff:.2e}  "
              f"{'✅' if max_diff < 1e-5 else '❌'}")

    # ---- 多步验证 ----
    p_t2 = p0.clone()
    m_t2 = m0.clone()
    v_t2 = v0.clone()
    p_r2 = p0.clone()
    m_r2 = m0.clone()
    v_r2 = v0.clone()

    for step in range(1, 11):
        g2 = torch.randn(N, device="cuda") * 0.1
        adamw_step(p_t2, g2, m_t2, v_t2, step=step)
        adamw_pytorch_step(p_r2, g2, m_r2, v_r2, step=step)

    max_diff_10 = (p_t2 - p_r2).abs().max().item()
    print(f"  [10-step] p max_diff={max_diff_10:.2e}  "
          f"{'✅' if max_diff_10 < 1e-5 else '❌'}")

    # ---- 性能对比 ----
    print("\n--- Performance (large tensor, 16M params) ---")
    N = 16777216
    p = torch.randn(N, device="cuda", dtype=torch.float32)
    grad = torch.randn(N, device="cuda", dtype=torch.float32)
    m = torch.randn(N, device="cuda", dtype=torch.float32)
    v = torch.randn(N, device="cuda", dtype=torch.float32).abs()

    def triton_step():
        pc, gc, mc, vc = p.clone(), grad.clone(), m.clone(), v.clone()
        adamw_step(pc, gc, mc, vc, step=100)

    def torch_step():
        pc, gc, mc, vc = p.clone(), grad.clone(), m.clone(), v.clone()
        adamw_pytorch_step(pc, gc, mc, vc, step=100)

    flops = N * 15  # ~15 FLOPs per element
    bw = N * 4 * 4 * 2  # p, g, m, v read + p, m, v write

    result = bench_compare({
        "Triton Fused (ours)": triton_step,
        "PyTorch Unfused": torch_step,
    }, flops=flops, bytes_accessed=bw, dtype="fp32")
    print_compare_report(result)


# PERFORMANCE NOTES
# =================
# - AdamW fusion 是实际训练中加速效果最大的优化之一:
#   - 将 6 个 kernel 合并为 1 个 → 减少 5× kernel launch overhead
#   - 避免 5 次 HBM round-trip (m_old→m_new, v_old→v_new, m̂, v̂, p_update)
#   - 实际加速 2-3x (取决于参数规模和 GPU)
# - 为什么 PyTorch 不自动做这个 fusion?
#   - PyTorch 2.0 的 torch.compile 可以自动融合 optimizer (部分场景)
#   - Triton 手写 kernel 可以保证最优的寄存器分配和指令调度
# - 为什么所有 constexpr 标量?
#   - lr, betas, eps, wd 是超参数, 在训练中不变
#   - 作为立即数嵌入 → 减少寄存器压力 → 更高 occupancy
# - LLM 训练中的应用:
#   - LLaMA, GPT 训练通常有数亿参数
#   - AdamW 占总训练时间的 5-15%
#   - 2x optimizer 加速 → 总体 5-10% 训练加速

if __name__ == "__main__":
    main()
