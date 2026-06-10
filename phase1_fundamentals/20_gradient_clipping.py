"""
20_gradient_clipping.py — Gradient Clipping (by global norm) kernel

学习目标：
  - 理解 gradient clipping 在训练稳定性中的作用
  - 掌握 "全局 reduction + elementwise scale" 两阶段模式
  - 学习 Triton 中如何做跨 program 的 reduction (atomic_add)

算法:
  给定梯度张量列表 grads = [g₁, g₂, ...]:
    total_norm = sqrt(sum(||gᵢ||²))
    if total_norm > max_norm:
      scale = max_norm / total_norm
      for g in grads: g *= scale

为什么需要 gradient clipping:
  - 防止梯度爆炸 (gradient explosion): 训练 RNN/Transformer 时常见
  - 稳定训练: 限制每次更新的步长
  - LLM 训练标配: GPT, Llama 等都用 max_norm=1.0

Triton 实现分两阶段:
  Stage 1: 每个 program 计算局部的 sum(g²), 用 atomic_add 累加到全局 total_norm²
  Stage 2: 如果超阈值, 每个元素乘以缩放因子

运行: python phase1_fundamentals/20_gradient_clipping.py
"""

import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.profiler import bench_compare, print_compare_report


@triton.jit
def compute_norm_sq_kernel(
    x_ptr, global_norm_sq_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Stage 1: 计算每个 program 的局部 sum(x²), 用 atomic_add 累加到全局.

    global_norm_sq_ptr 指向一个 scalar, 所有 program 对其做 atomic_add.
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    local_sq_sum = tl.sum(x * x, axis=0)  # 标量: 当前 block 的 sum(x²)

    # 原子累加到全局 (跨所有 program)
    tl.atomic_add(global_norm_sq_ptr, local_sq_sum)


@triton.jit
def scale_kernel(
    x_ptr, output_ptr,
    scale,  # 缩放因子 (scalar, 所有元素共用)
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Stage 2: elementwise scale."""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    output = x * scale
    tl.store(output_ptr + offsets, output, mask=mask)


def clip_grad_norm(
    grads: list[torch.Tensor], max_norm: float
) -> torch.Tensor:
    """
    Gradient clipping by global norm.

    Args:
        grads: 梯度张量列表
        max_norm: 最大允许的 norm

    Returns:
        total_norm: 裁剪前的总 norm (用于 logging)
    """
    # Stage 1: 计算 total_norm² = sum(sum(g²) for g in grads)
    total_norm_sq = torch.zeros(1, device=grads[0].device, dtype=torch.float32)

    for g in grads:
        if g is None:
            continue
        n = g.numel()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        compute_norm_sq_kernel[grid](g, total_norm_sq, n, BLOCK_SIZE=1024)

    total_norm = torch.sqrt(total_norm_sq)

    # Stage 2: 如果超过阈值, 缩放所有梯度
    if total_norm > max_norm:
        scale = (max_norm / total_norm).item()  # Python float for Triton scalar
        for g in grads:
            if g is None:
                continue
            n = g.numel()
            grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
            scale_kernel[grid](g, g, scale, n, BLOCK_SIZE=1024)

    return total_norm


def main():
    print("=" * 60)
    print("20_gradient_clipping — Gradient Clipping by Norm")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    torch.manual_seed(42)

    # 模拟多个梯度张量
    grads = [
        torch.randn(1024, 4096, device="cuda") * 5,
        torch.randn(512, 256, device="cuda") * 10,
        torch.randn(65536, device="cuda") * 2,
    ]

    # ---- 正确性 ----
    # PyTorch reference
    total_norm_torch = torch.nn.utils.clip_grad_norm_(
        [g.clone() for g in grads], max_norm=1.0
    )

    # Our implementation
    grads_copy = [g.clone() for g in grads]
    total_norm_triton = clip_grad_norm(grads_copy, max_norm=1.0)

    norm_diff = abs(total_norm_triton.item() - total_norm_torch.item())
    print(f"  total_norm: triton={total_norm_triton.item():.4f}  "
          f"torch={total_norm_torch.item():.4f}  diff={norm_diff:.2e}  "
          f"{'✅' if norm_diff < 1e-4 else '❌'}")

    # 验证裁剪结果一致
    grads_ref = [g.clone() for g in grads]
    torch.nn.utils.clip_grad_norm_(grads_ref, max_norm=1.0)
    all_close = True
    for i, (gt, gr) in enumerate(zip(grads_copy, grads_ref)):
        diff = (gt - gr).abs().max().item()
        ok = diff < 1e-4
        if not ok:
            all_close = False
        print(f"  grad[{i}]: max_diff={diff:.2e}  {'✅' if ok else '❌'}")

    # ---- 性能对比 ----
    print("\n--- Performance (single large tensor) ---")
    g = torch.randn(16777216, device="cuda", dtype=torch.float32) * 5
    n = g.numel()

    def triton_clip():
        gg = [g.clone()]
        clip_grad_norm(gg, 1.0)

    def torch_clip():
        gg = [g.clone()]
        torch.nn.utils.clip_grad_norm_(gg, 1.0)

    result = bench_compare({
        "Triton (ours)": triton_clip,
        "PyTorch (ref)": torch_clip,
    }, flops=n * 3, bytes_accessed=n * 2 * 4, dtype="fp32")
    print_compare_report(result)


# PERFORMANCE NOTES
# =================
# - Gradient clipping 分两阶段:
#   1. norm reduction: O(N) 读, atomic_add 写 (memory-bound)
#   2. elementwise scale: O(N) 读写 (memory-bound, 只在超过阈值时执行)
# - tl.atomic_add 的性能:
#   - 对 global memory 的原子操作有竞争开销
#   - 对 H100: atomic_add 延迟 ~100 cycles (vs regular add ~4 cycles)
#   - 实践中 grid 通常 ~100-1000 programs → atomic contention 可控
# - PyTorch 的 clip_grad_norm_ 使用多个 CUDA kernel (norm + scale)
# - Triton 的优势: 可以进一步融合 norm + scale 为一个 kernel (fused backward)

if __name__ == "__main__":
    main()
