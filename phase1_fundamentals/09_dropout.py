"""
08_dropout.py — Dropout kernel with Philox RNG

学习目标：
  - 理解 Dropout 的正向和反向计算
  - 掌握 Triton 中随机数生成 (Philox RNG via tl.rand)
  - 学习 fused dropout + scale 的好处

数学公式:
  训练时: output = x * mask / (1-p),  where mask_i ∈ {0, 1}, P(mask_i=1) = 1-p
  推理时: output = x  (无操作)

  "Inverted Dropout":
    除以 (1-p) 使得训练和推理时的期望值一致: E[output] = E[x]
    推理时无需缩放 → 简化部署

随机数生成:
  Triton 使用 Philox 算法生成伪随机数:
    - Philox: counter-based PRNG, 基于 AES 轮函数
    - 每个 program 有独立的 seed (base_seed + program_id)
    - 确定性: 相同 seed + offset → 相同随机数

运行: python phase1_fundamentals/09_dropout.py
"""

import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.profiler import bench_compare, print_compare_report


@triton.jit
def dropout_kernel(
    x_ptr,       # 输入指针
    output_ptr,  # 输出指针
    mask_ptr,    # mask 输出 (用于反向传播)
    n_elements,
    p: tl.constexpr,          # 丢弃概率 (0 ≤ p < 1)
    seed,                     # 随机种子 (int32 scalar)
    BLOCK_SIZE: tl.constexpr,
):
    """
    Inverted Dropout: output = x * mask / (1-p), mask ∈ {0, 1}

    每个 program 有独立的随机数流 (seed = base_seed + program_id)。
    Philox RNG 保证不同 program 间的随机数不相关。
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # 加载输入
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # ---- 生成随机 mask ----
    # Philox RNG: 每个 program 使用 seed + pid 作为随机种子
    # offset 用于区分同一 program 内的不同元素
    # [COMPILER] tl.rand 生成 uniform [0, 1) 随机数
    rand_vals = tl.rand(seed + pid, offsets)

    # mask: 保留概率 = 1-p, 丢弃概率 = p
    # rand > p → 保留 (1), rand ≤ p → 丢弃 (0)
    keep_mask = rand_vals > p

    # Inverted dropout: 除以 (1-p) 保持期望值不变
    # [COMPILER] tl.where 编译为 select 指令 (无分支)
    scale: tl.constexpr = 1.0 / (1.0 - p)
    output = tl.where(keep_mask, x * scale, 0.0)

    # 写回
    tl.store(output_ptr + offsets, output, mask=mask)
    tl.store(mask_ptr + offsets, keep_mask.to(tl.float32), mask=mask)


def dropout(x: torch.Tensor, p: float = 0.5,
            seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Inverted Dropout 包装函数。

    Args:
        x: 输入张量
        p: 丢弃概率 (0 ≤ p < 1), 默认 0.5
        seed: 随机种子 (用于确定性测试或不同的随机模式)

    Returns:
        (output, mask) — output 是 dropout 后的结果, mask 用于反向传播
    """
    output = torch.empty_like(x)
    mask = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    dropout_kernel[grid](x, output, mask, n_elements, p=p, seed=seed, BLOCK_SIZE=1024)
    return output, mask


def test_correctness():
    """验证 dropout 的统计特性 (非逐元素对比, 因为 PyTorch 的 mask 也不同)。"""
    torch.manual_seed(42)

    shape = (32, 1024)
    p = 0.3
    x = torch.ones(shape, device="cuda")  # 全 1 → 容易验证统计特性

    # 测试 1: 我们的实现内部一致性 (相同 seed → 相同结果)
    out1, mask1 = dropout(x, p=p, seed=12345)
    out2, mask2 = dropout(x, p=p, seed=12345)
    max_diff = (out1 - out2).abs().max().item()
    print(f"  Determinism (same seed): max_diff={max_diff:.0e}  "
          f"{'✅' if max_diff == 0 else '❌'}")

    # 测试 2: 不同 seed → 不同 mask
    out3, mask3 = dropout(x, p=p, seed=54321)
    same_mask = (mask1 == mask3).float().mean().item()
    print(f"  Different masks (diff seed): same_frac={same_mask:.3f}  "
          f"{'✅' if 0.6 < same_mask < 0.8 else '⚠️'}  (expect ~0.7)")

    # 测试 3: 保留比例验证
    keep_frac = mask1.float().mean().item()
    expected_keep = 1.0 - p
    print(f"  Keep fraction: {keep_frac:.4f} (expect {expected_keep:.4f})  "
          f"{'✅' if abs(keep_frac - expected_keep) < 0.02 else '❌'}")

    # 测试 4: 输出缩放验证 — 保留的元素值应该 = 1/(1-p)
    kept_values = out1[mask1 > 0.5]
    expected_val = 1.0 / (1.0 - p)
    val_diff = (kept_values - expected_val).abs().max().item()
    print(f"  Scale check: kept_val={kept_values[0].item():.4f} "
          f"(expect {expected_val:.4f})  "
          f"{'✅' if val_diff < 1e-5 else '❌'}")

    # 测试 5: 丢弃的元素值应该 = 0
    dropped_values = out1[mask1 < 0.5]
    if len(dropped_values) > 0:
        dropped_max = dropped_values.abs().max().item()
        print(f"  Dropped are zero: max_abs={dropped_max:.0e}  "
              f"{'✅' if dropped_max == 0 else '❌'}")


def main():
    print("=" * 60)
    print("08_dropout — Dropout with Philox RNG")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # ---- 统计正确性测试 ----
    print("--- Correctness (statistical) ---")
    test_correctness()

    # ---- 性能对比: Triton vs PyTorch ----
    print("\n--- Performance ---")

    p = 0.5
    x = torch.randn(16777216, device="cuda", dtype=torch.float32)  # 16M

    # PyTorch dropout (training mode)
    def torch_dropout():
        return torch.nn.functional.dropout(x, p=p, training=True)

    n_elements = x.numel()
    # rand generation (~2) + compare (1) + conditional mul (1) ≈ 4 FLOPs
    flops_total = n_elements * 4
    # x(4B) read + out(4B) write + mask(4B) write
    bytes_total = n_elements * 3 * 4

    result = bench_compare(
        {
            "Triton (ours)": lambda: dropout(x, p=p, seed=torch.randint(0, 2**31-1, (1,)).item())[0],
            "PyTorch (ref)": torch_dropout,
        },
        flops=flops_total,
        bytes_accessed=bytes_total,
        dtype="fp32",
    )
    print_compare_report(result)


# PERFORMANCE NOTES
# =================
# - Dropout 是 memory-bound: ~4 FLOPs / 12 bytes ≈ 0.33 FLOP/byte
# - Philox RNG 的开销:
#   - 每个元素需要 2-3 个 Philox round (AES 轮函数)
#   - 比简单数学运算贵, 但和 HBM 访问相比仍然很小
# - Fused dropout (mask + scale + store) 的优势:
#   - 如果不 fusion: 生成 mask (write) → 读回 mask → apply (read+write)
#     多了 2 次 HBM round-trip → ~2x 带宽消耗
# - [COMPILER] tl.where 编译为 LLVM select, 无分支 → 无 warp divergence
# - 实际优化方向:
#   - Fused dropout + previous op (如 dropout(gelu(x)))
#   - 使用 tensor core 的随机数生成 (Hopper+)
# - Dropout 在推理时完全不需要, 但在训练中至关重要
#   - 推理时 dropout(..., training=False) 直接返回 x → 零开销

if __name__ == "__main__":
    main()
