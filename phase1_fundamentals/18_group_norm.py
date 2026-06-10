"""
13_group_norm.py — Group Normalization kernel

学习目标：
  - 理解 GroupNorm vs LayerNorm vs BatchNorm 的差异
  - 掌握分组 reduction 的 indexing 模式
  - 学习如何对不连续内存做分块迭代

数学公式:
  GroupNorm(x; γ, β) = γ * (x - μ_g) / √(σ²_g + ε) + β

  其中 μ_g 和 σ²_g 是在同一 group 内计算的:
    - 将 C 个 channel 分为 G 个 group, 每组有 C/G 个 channel
    - 对每个 group 独立计算 mean 和 variance
    - γ 和 β 是 per-channel 的可学习参数

归一化维度对比:
  - BatchNorm:   跨 (N, H, W),  对每个 channel  (最常见于 CNN)
  - LayerNorm:    跨 (C, H, W),  对每个 sample    (最常见于 Transformer)
  - InstanceNorm: 跨 (H, W),     对每个 channel×sample (风格迁移)
  - GroupNorm:    跨 (C/G, H, W), 对每个 group×sample (介于 LN 和 IN 之间)

使用场景:
  - 小 batch 训练: BatchNorm 在小 batch 下不稳定 → GroupNorm 替代
  - 检测/分割: Mask R-CNN, YOLO 等使用 GroupNorm
  - 扩散模型: Stable Diffusion 使用 GroupNorm
  - G = C → InstanceNorm; G = 1 → LayerNorm (对 2D 输入)

运行: python phase1_fundamentals/18_group_norm.py
"""

import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.profiler import bench_compare, print_compare_report


@triton.jit
def group_norm_kernel(
    x_ptr,        # 输入: (N * spatial_size, C) — 展平空间维度
    weight_ptr,   # γ (scale), shape: (C,)
    bias_ptr,     # β (shift), shape: (C,)
    output_ptr,   # 输出: 同 shape
    n_cols,       # C (总通道数)
    spatial_size, # 每个 sample 的空间位置数 (H*W*...)
    channels_per_group: tl.constexpr,  # C // G
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Group Normalization: 每个 program 处理一个 (sample, group) 对,
    统计量在 C/G 个通道 × spatial_size 个空间位置上计算。

    内存布局: x[row][col], row_idx = sample * spatial_size + spatial_idx
    每个 sample 占据 contiguous 的 spatial_size 行。
    """
    pid = tl.program_id(axis=0)

    n_groups = n_cols // channels_per_group
    sample_idx = pid // n_groups
    group_idx = pid % n_groups

    # 当前 group 在 C 维度上的范围
    group_start = group_idx * channels_per_group

    # 当前 sample 的行范围
    row_start = sample_idx * spatial_size
    row_end = row_start + spatial_size

    col_offsets = tl.arange(0, BLOCK_SIZE)

    # ---- Pass 1: 计算 mean —— 遍历所有空间位置和组内通道 ----
    accum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    n_elements_per_group = 0  # 实际处理的有效元素数
    for row_idx in range(row_start, row_end):
        row_base = row_idx * n_cols
        for block_start in range(0, channels_per_group, BLOCK_SIZE):
            offsets = row_base + group_start + block_start + col_offsets
            mask = (block_start + col_offsets) < channels_per_group
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            accum += x
            n_elements_per_group += channels_per_group  # 每次 block 迭代的有效元素

    total_elements = spatial_size * channels_per_group
    group_mean = tl.sum(accum, axis=0) / total_elements

    # ---- Pass 2: 计算 variance —— 同样遍历所有位置 ----
    sq_accum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for row_idx in range(row_start, row_end):
        row_base = row_idx * n_cols
        for block_start in range(0, channels_per_group, BLOCK_SIZE):
            offsets = row_base + group_start + block_start + col_offsets
            mask = (block_start + col_offsets) < channels_per_group
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            diff = x - group_mean
            # [FIX]: 只累加有效元素的平方差, masked 元素贡献 0
            sq_accum += tl.where(mask, diff * diff, 0.0)
    group_var = tl.sum(sq_accum, axis=0) / total_elements

    # ---- Pass 3: 归一化 + affine + 写回 ----
    inv_std = 1.0 / tl.sqrt(group_var + eps)
    for row_idx in range(row_start, row_end):
        row_base = row_idx * n_cols
        for block_start in range(0, channels_per_group, BLOCK_SIZE):
            offsets = row_base + group_start + block_start + col_offsets
            mask = (block_start + col_offsets) < channels_per_group
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

            w = tl.load(weight_ptr + group_start + block_start + col_offsets,
                        mask=mask, other=0.0)
            b = tl.load(bias_ptr + group_start + block_start + col_offsets,
                        mask=mask, other=0.0)

            normalized = w * (x - group_mean) * inv_std + b
            tl.store(output_ptr + offsets, normalized, mask=mask)


def group_norm(
    x: torch.Tensor,
    num_groups: int,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Group Normalization。

    Args:
        x: 输入 (N, C, ...), 如 (N, C) 或 (N, C, H, W)
        num_groups: group 数量 G
        weight: scale γ, shape (C,)
        bias: shift β, shape (C,)
        eps: 数值稳定常数

    Returns:
        归一化后的张量, shape 与 x 相同
    """
    assert x.dim() >= 2, f"Expected at least 2D input, got {x.dim()}D"
    C = x.shape[1]
    assert C % num_groups == 0, f"C={C} must be divisible by num_groups={num_groups}"
    assert weight.shape == (C,), f"weight shape {weight.shape} != ({C},)"
    assert bias.shape == (C,), f"bias shape {bias.shape} != ({C},)"

    orig_shape = x.shape
    channels_per_group = C // num_groups

    # Reshape: (N, C, ...) → (N, C, S) where S = all spatial dims
    # Then transpose to (N, S, C) and flatten to (N*S, C)
    N = x.shape[0]
    spatial_dims = x.shape[2:] if x.dim() > 2 else (1,)
    spatial_size = 1
    for d in spatial_dims:
        spatial_size *= d

    # (N, C, S) → (N, S, C) → (N*S, C)
    if x.dim() > 2:
        x_reshaped = x.reshape(N, C, spatial_size).permute(0, 2, 1).reshape(-1, C)
    else:
        x_reshaped = x.reshape(-1, C)
        spatial_size = 1  # 2D input has no spatial dims

    output_flat = torch.empty_like(x_reshaped)

    # grid: N samples × G groups
    grid = (N * num_groups,)
    group_norm_kernel[grid](
        x_reshaped, weight, bias, output_flat,
        C, spatial_size, channels_per_group=channels_per_group,
        eps=eps, BLOCK_SIZE=256,
    )

    # Reshape output back to original shape
    if len(orig_shape) > 2:
        output_flat = output_flat.reshape(N, spatial_size, C).permute(0, 2, 1).reshape(orig_shape)
    else:
        output_flat = output_flat.reshape(orig_shape)

    return output_flat


def main():
    print("=" * 60)
    print("13_group_norm — Group Normalization")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # ---- 正确性测试 ----
    torch.manual_seed(42)

    test_cases = [
        # (name, shape, num_groups)
        ("2D-G4", (256, 64), 4),
        ("2D-G8", (512, 128), 8),
        ("4D-G8", (2, 32, 64, 64), 8),    # CNN feature map
        ("4D-G16", (4, 64, 32, 32), 16),   # larger
        # 边界情况
        ("G=1=LN", (128, 256), 1),          # G=1 → 等价于 LayerNorm (对2D)
        ("G=C=IN", (64, 32), 32),           # G=C → 等价于 InstanceNorm (对2D)
    ]

    for name, shape, num_groups in test_cases:
        x = torch.randn(*shape, device="cuda")
        C = shape[1]
        weight = torch.randn(C, device="cuda")
        bias = torch.randn(C, device="cuda")

        out_triton = group_norm(x, num_groups, weight, bias)
        out_torch = torch.nn.functional.group_norm(
            x, num_groups, weight, bias, eps=1e-5
        )

        max_diff = (out_triton - out_torch).abs().max().item()
        flat_diff = (out_triton.reshape(-1) - out_torch.reshape(-1)).abs()
        mean_diff = flat_diff.mean().item()
        status = "✅" if max_diff < 1e-3 else "❌"
        print(f"  [{name:>8s}] shape={list(shape)}  "
              f"max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}  {status}")

    # ---- 性能对比 (使用 2D 输入避免空间维度的循环开销) ----
    print("\n--- Performance (2D input, no spatial dims) ---")

    x = torch.randn(4096, 1024, device="cuda", dtype=torch.float32)
    num_groups = 32
    weight = torch.randn(1024, device="cuda")
    bias = torch.randn(1024, device="cuda")

    n_elements = x.numel()
    flops_total = n_elements * 8
    bytes_total = n_elements * 4 * 4

    result = bench_compare(
        {
            "Triton (ours)": lambda: group_norm(x, num_groups, weight, bias),
            "PyTorch (ref)": lambda: torch.nn.functional.group_norm(
                x, num_groups, weight, bias, eps=1e-5
            ),
        },
        flops=flops_total,
        bytes_accessed=bytes_total,
        dtype="fp32",
    )
    print_compare_report(result)


# PERFORMANCE NOTES
# =================
# - GroupNorm 是 memory-bound: ~8 FLOPs / 16 bytes = 0.5 FLOP/byte
# - 和 LayerNorm 的主要区别:
#   - LayerNorm: 每个 sample 一个 program, 处理 C 个元素 (1D reduction)
#   - GroupNorm: 每个 (sample, group) 一个 program, 处理 C/G 个元素
#   - GroupNorm 的 grid 更大 (N*G vs N) → 更多 parallelism
# - 归一化族谱:
#   - G=1:    GroupNorm ≈ LayerNorm (但 LayerNorm 通常没有 spatial 维度)
#   - G=C:    GroupNorm ≈ InstanceNorm
#   - 1<G<C:  GroupNorm (中间地带)
# - 为什么小 batch 用 GroupNorm 而不是 BatchNorm:
#   - BatchNorm 统计量跨 batch 计算, 小 batch 时不稳定
#   - GroupNorm 统计量在单个 sample 内计算, 与 batch size 无关
# - 本实现的局限性 (3-pass):
#   - 和 LayerNorm 一样, 读 x 3 次
#   - 可以优化为 2-pass 或 Welford 1-pass
# - 本实现的局限性:
#   1. 3-pass 算法读 x 三次 (可优化为 Welford 1-pass)
#   2. 对于 4D 输入 (N, C, H, W), 每个 (sample, group) program 需要遍历
#      所有空间位置, 当 H*W 很大时 (如 128×128=16384), 循环开销巨大
#      → 生产实现应用 shared memory 缓存空间块, 或做 hierarchical reduction
#   3. 性能测试使用 2D 输入 (无空间维度) 以避免循环开销
# - 对于 4D 输入, PyTorch cuDNN 的实现快 40x+

if __name__ == "__main__":
    main()
