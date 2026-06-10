"""
11_cross_entropy.py — Fused Cross Entropy Loss kernel

学习目标：
  - 理解 Cross Entropy Loss 的数学和数值稳定实现
  - 掌握 log_softmax 的 online 算法（max subtraction trick）
  - 学习 Triton 中的 reduction + gather 组合模式

数学公式:
  CrossEntropy(logits, target) = -mean(log(softmax(logits[i, target[i]])))

  数值稳定的计算:
    1. max_val = max(logits, dim=-1)  — 减去最大值防止 exp 溢出
    2. softmax = exp(logits - max_val) / sum(exp(logits - max_val))
    3. log_softmax = (logits - max_val) - log(sum(exp(logits - max_val)))
    4. loss = -log_softmax[target]

Max subtraction trick:
  为什么安全?  softmax(x_i) = exp(x_i - max) / Σexp(x_j - max)
  分子分母同除 exp(max), 数学上等价, 但 exp(x_i - max) ≤ exp(0) = 1, 不会溢出

在深度学习中的使用:
  - 分类任务的标准损失函数
  - LLM 训练: 每个 token 都是一个分类问题 (vocab_size 类)
  - 和 softmax 一样是 memory-bound

对比:
  - PyTorch: F.cross_entropy (cuDNN backend, 高度优化)
  - Liger Kernel: liger_cross_entropy (fused Triton 实现)

运行: python phase1_fundamentals/18_cross_entropy.py
"""

import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.profiler import bench_compare, print_compare_report
from benchmarks.references.liger_ref import get_liger_cross_entropy


@triton.jit
def cross_entropy_kernel(
    logits_ptr,   # 输入 logits: (N_ROWS, N_COLS)
    labels_ptr,   # 目标标签: (N_ROWS,), 每个值在 [0, N_COLS-1]
    loss_ptr,     # 输出 per-row loss: (N_ROWS,)
    n_cols,
    ignore_index: tl.constexpr,  # 忽略的标签值 (如 -100), 对应的行 loss=0
    BLOCK_SIZE: tl.constexpr,
):
    """
    逐行计算 cross entropy loss: loss[i] = -log_softmax(logits[i, labels[i]])

    每个 program 处理一行, 2-pass 算法。
    """
    row_idx = tl.program_id(axis=0)
    row_start = row_idx * n_cols
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # 加载该行的标签
    label = tl.load(labels_ptr + row_idx)

    # ---- Pass 1: 找每行最大值（数值稳定性） ----
    row_max = tl.full([BLOCK_SIZE], float("-inf"), dtype=tl.float32)
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = row_start + block_start + col_offsets
        mask = (block_start + col_offsets) < n_cols
        logits = tl.load(logits_ptr + offsets, mask=mask, other=float("-inf"))
        row_max = tl.maximum(row_max, logits)
    global_max = tl.max(row_max, axis=0)

    # ---- Pass 2: 计算 sum(exp(logits - max)) ----
    sum_exp = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = row_start + block_start + col_offsets
        mask = (block_start + col_offsets) < n_cols
        logits = tl.load(logits_ptr + offsets, mask=mask, other=float("-inf"))
        sum_exp += tl.exp(logits - global_max)
    global_sum_exp = tl.sum(sum_exp, axis=0)

    # ---- Compute loss at target position ----
    # log_softmax[label] = logits[label] - max - log(sum_exp)
    # loss = -log_softmax[label]

    # 需要从 logits 中取出 label 位置的值
    # 策略: 遍历所有列, 在 label 位置处计算 loss
    loss_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = row_start + block_start + col_offsets
        mask = (block_start + col_offsets) < n_cols
        logits = tl.load(logits_ptr + offsets, mask=mask, other=0.0)

        # 找到 label 所在的位置: 只在该位置计算 loss
        col_indices = block_start + col_offsets
        is_target = (col_indices == label) & mask
        # log_softmax = logits - max - log(sum_exp)
        # 使用安全的 log: 如果 sum_exp == 0 则 log 会出错, 但 softmax 定义保证 > 0
        log_prob = logits - global_max - tl.log(global_sum_exp)
        loss_val = tl.where(is_target, -log_prob, loss_val)

    # 处理 ignore_index: 如果标签 == ignore_index, loss = 0
    global_loss = tl.sum(loss_val, axis=0)
    is_ignored = (label == ignore_index)
    final_loss = tl.where(is_ignored, 0.0, global_loss)

    tl.store(loss_ptr + row_idx, final_loss)


def cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Cross Entropy Loss (fused log_softmax + nll_loss).

    Args:
        logits: 预测 logits (N, C)
        labels: 目标标签 (N,), 每个值在 [0, C-1]
        ignore_index: 忽略的标签索引 (默认 -100)
        reduction: 'mean', 'sum', 或 'none'

    Returns:
        loss: scalar if reduction in ('mean', 'sum'), else (N,)
    """
    assert logits.dim() == 2, f"Expected 2D logits, got {logits.dim()}D"
    assert labels.dim() == 1, f"Expected 1D labels, got {labels.dim()}D"
    assert logits.shape[0] == labels.shape[0]

    n_rows, n_cols = logits.shape
    per_row_loss = torch.empty(n_rows, device=logits.device, dtype=torch.float32)

    grid = (n_rows,)  # 每行一个 program
    cross_entropy_kernel[grid](
        logits, labels, per_row_loss, n_cols,
        ignore_index=ignore_index, BLOCK_SIZE=1024,
    )

    if reduction == "none":
        return per_row_loss
    elif reduction == "sum":
        return per_row_loss.sum()
    else:  # mean
        # 排除 ignore_index 的行
        valid_mask = (labels != ignore_index)
        n_valid = valid_mask.sum().clamp(min=1)
        return per_row_loss.sum() / n_valid


def main():
    print("=" * 60)
    print("11_cross_entropy — Fused Cross Entropy Loss")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # ---- 正确性测试 ----
    torch.manual_seed(42)

    test_cases = [
        # (name, n_rows, n_classes)
        ("small", 128, 256),
        ("medium", 512, 4096),
        ("large-vocab", 256, 32000),  # 模拟语言模型的词表大小
    ]

    for name, n_rows, n_classes in test_cases:
        logits = torch.randn(n_rows, n_classes, device="cuda")
        labels = torch.randint(0, n_classes, (n_rows,), device="cuda")

        out_triton = cross_entropy_loss(logits, labels, ignore_index=-100, reduction="mean")
        out_torch = torch.nn.functional.cross_entropy(
            logits, labels, ignore_index=-100, reduction="mean"
        )

        diff = (out_triton - out_torch).abs().item()
        status = "✅" if diff < 1e-3 else "❌"
        print(f"  [{name:>12s}] ({n_rows}, {n_classes})  "
              f"loss_triton={out_triton.item():.4f}  loss_torch={out_torch.item():.4f}  "
              f"diff={diff:.2e}  {status}")

    # ---- ignore_index 测试 ----
    logits = torch.randn(64, 128, device="cuda")
    labels = torch.randint(0, 128, (64,), device="cuda")
    labels[10:20] = -100  # 忽略 10 个样本

    out_triton = cross_entropy_loss(logits, labels, ignore_index=-100, reduction="mean")
    out_torch = torch.nn.functional.cross_entropy(
        logits, labels, ignore_index=-100, reduction="mean"
    )
    diff = (out_triton - out_torch).abs().item()
    print(f"  [ignore_idx] loss_triton={out_triton.item():.4f}  "
          f"loss_torch={out_torch.item():.4f}  diff={diff:.2e}  "
          f"{'✅' if diff < 1e-3 else '❌'}")

    # ---- 性能对比: Triton vs PyTorch vs Liger ----
    print("\n--- Performance ---")

    # 模拟典型 LLM 场景: batch*seq ~2048, vocab ~32000
    logits = torch.randn(2048, 32000, device="cuda", dtype=torch.float32)
    labels = torch.randint(0, 32000, (2048,), device="cuda")

    implementations = {
        "Triton (ours)": lambda: cross_entropy_loss(logits, labels),
        "PyTorch (ref)": lambda: torch.nn.functional.cross_entropy(logits, labels),
    }

    # Liger cross entropy
    liger_ce = get_liger_cross_entropy()
    if liger_ce:
        implementations["Liger (SotA)"] = lambda: liger_ce(logits, labels)

    n_elements = logits.numel()
    # exp(~4) + sub(1) + max(1) + sum(1) + log(~4) + sub(1) + neg(1) ≈ 13 FLOPs per element
    flops_total = n_elements * 13
    # logits read(4B) + labels read(~8B per row, negligible) + loss write(4B per row)
    bytes_total = n_elements * 4 + labels.numel() * 8 + logits.shape[0] * 4

    result = bench_compare(
        implementations,
        flops=flops_total,
        bytes_accessed=bytes_total,
        dtype="fp32",
    )
    print_compare_report(result)


# PERFORMANCE NOTES
# =================
# - Cross Entropy 是 memory-bound: ~13 FLOPs / ~4 bytes ≈ 3.3 FLOP/byte
#   (每个元素大约只读一次 → 算术强度比 softmax 高, 但仍然 memory-bound)
# - 数值稳定性的关键:
#   1. max subtraction: 防止 exp 溢出 (exp(88) ≈ 1.6e38 > fp32 max)
#   2. log(sum_exp) 而非 sum(log): 避免 log(0) = -inf
# - online 算法: 2-pass (max → sum_exp), 不需要存中间结果到 HBM
# - LLM 中的特点:
#   - vocab_size 通常 32000-128000 → 行很长 → 需要分块迭代
#   - 每行做一次 max reduction + sum reduction → 类似 softmax 但多一次 gather
# - 和 PyTorch 的差距:
#   - PyTorch 使用 cuDNN 的 softmax + nll_loss (可能用 tensor core)
#   - 我们的实现是 2-pass (读 logits 2次), 可优化为 1-pass Welford 风格
# - Liger 的 cross_entropy 也是 Triton 实现, 但做了更多优化:
#   - Fused backward (保存 softmax 结果供反向传播复用)
#   - 更好的 shared memory 使用

if __name__ == "__main__":
    main()
