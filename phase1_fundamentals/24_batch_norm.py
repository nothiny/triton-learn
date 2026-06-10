"""
24_batch_norm.py — Batch Normalization 1D kernel

学习目标：
  - 理解 BatchNorm vs LayerNorm 的根本差异 (跨样本 vs 样本内)
  - 掌握跨 sample strided reduction 的 kernel 设计
  - 理解非 coalesced 访存的性能影响

数学公式:
  y = γ * (x - μ) / √(σ² + ε) + β
  其中 μ, σ² 在当前 mini-batch 上沿 N 维度计算 (每个 channel 独立)

归一化维度对比 (以 N,C 输入为例):
  BatchNorm:   跨 N,      每个 C 独立  → grid = C
  LayerNorm:    跨 C,      每个 N 独立  → grid = N
  GroupNorm:    跨 C/G,    每个 (N,G)  → grid = N*G

性能挑战:
  - 输入 (N, C) 是 row-major: 同一 channel 的不同 sample 间隔 C 个元素
  - 非 coalesced 访存 → 带宽利用率大幅下降
  - 这也是为什么 Transformer 不用 BatchNorm 的原因之一

运行: python phase1_fundamentals/24_batch_norm.py
"""

import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.profiler import bench_compare, print_compare_report


@triton.jit
def batchnorm1d_kernel(
    x_ptr,        # (N, C) row-major
    weight_ptr,   # γ (C,)
    bias_ptr,     # β (C,)
    output_ptr,   # (N, C)
    n_samples,    # N
    n_cols,       # C
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    每个 program 处理一个 channel, 沿 N 维度做 strided reduction.
    (教学实现 — strided 访存性能受限)
    """
    cid = tl.program_id(axis=0)  # channel index
    offsets_n = tl.arange(0, BLOCK_SIZE)

    # ---- Pass 1: mean (strided) ----
    accum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for block_start in range(0, n_samples, BLOCK_SIZE):
        # offset = row * n_cols + cid, where row = block_start + offsets_n
        row_indices = block_start + offsets_n
        offsets = row_indices * n_cols + cid
        mask = row_indices < n_samples
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        accum += x
    mean = tl.sum(accum, axis=0) / n_samples

    # ---- Pass 2: variance (strided) ----
    sq_accum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for block_start in range(0, n_samples, BLOCK_SIZE):
        row_indices = block_start + offsets_n
        offsets = row_indices * n_cols + cid
        mask = row_indices < n_samples
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        diff = x - mean
        sq_accum += tl.where(mask, diff * diff, 0.0)
    var = tl.sum(sq_accum, axis=0) / n_samples

    # ---- Pass 3: normalize (strided) ----
    inv_std = 1.0 / tl.sqrt(var + eps)
    w = tl.load(weight_ptr + cid)
    b = tl.load(bias_ptr + cid)

    for block_start in range(0, n_samples, BLOCK_SIZE):
        row_indices = block_start + offsets_n
        offsets = row_indices * n_cols + cid
        mask = row_indices < n_samples
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        normalized = w * (x - mean) * inv_std + b
        tl.store(output_ptr + offsets, normalized, mask=mask)


def batchnorm1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    BatchNorm1D (training mode).

    Args:
        x: (N, C) 或 (N, C, L) — 空间维度会被 flatten 到 N
        weight: γ (C,)
        bias: β (C,)
    """
    orig_shape = x.shape
    C = x.shape[1]
    N = x.numel() // C  # flatten all dims except C into N

    x_flat = x.reshape(N, C)
    output_flat = torch.empty_like(x_flat)

    grid = (C,)  # 每个 channel 一个 program
    batchnorm1d_kernel[grid](
        x_flat, weight, bias, output_flat,
        N, C, eps=eps, BLOCK_SIZE=256,
    )
    return output_flat.reshape(orig_shape)


def main():
    print("=" * 60)
    print("24_batch_norm — Batch Normalization 1D")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    torch.manual_seed(42)

    # ---- 正确性 ----
    N, C = 128, 64
    x = torch.randn(N, C, device="cuda")
    w = torch.randn(C, device="cuda")
    b = torch.randn(C, device="cuda")

    out_triton = batchnorm1d(x, w, b)
    out_torch = torch.nn.functional.batch_norm(
        x, None, None, weight=w, bias=b, training=True, eps=1e-5
    )

    max_diff = (out_triton - out_torch).abs().max().item()
    print(f"  shape=({N}, {C})  max_diff={max_diff:.2e}  "
          f"{'✅' if max_diff < 1e-4 else '❌'}")

    # ---- 跨 channel 验证: 每个 channel 的 output mean ≈ 0 ----
    channel_means = out_triton.mean(dim=0)
    mean_of_means = channel_means.abs().max().item()
    print(f"  per-channel mean (max abs): {mean_of_means:.2e}  "
          f"{'✅' if mean_of_means < 1e-4 else '❌'} "
          f"(should be ≈ 0)")

    # ---- 和 LayerNorm 对比 ----
    print("\n--- LayerNorm vs BatchNorm (same input) ---")
    x = torch.randn(64, 128, device="cuda")
    out_ln = torch.nn.functional.layer_norm(x, [128], torch.ones(128, device="cuda"),
                                             torch.zeros(128, device="cuda"))
    out_bn = batchnorm1d(x, torch.ones(128, device="cuda"), torch.zeros(128, device="cuda"))

    ln_sample_means = out_ln.mean(dim=1)
    bn_sample_means = out_bn.mean(dim=1)
    ln_channel_means = out_ln.mean(dim=0)
    bn_channel_means = out_bn.mean(dim=0)

    print(f"  LayerNorm: sample_means≈0? {ln_sample_means.abs().max().item():.2e}  "
          f"channel_means≈0? {ln_channel_means.abs().max().item():.2e}")
    print(f"  BatchNorm: sample_means≈0? {bn_sample_means.abs().max().item():.2e}  "
          f"channel_means≈0? {bn_channel_means.abs().max().item():.2e}")
    print(f"  Key: LN→per-sample zero-mean; BN→per-channel zero-mean")

    print("\n--- Performance ---")
    x = torch.randn(8192, 256, device="cuda", dtype=torch.float32)
    w = torch.randn(256, device="cuda")
    b = torch.randn(256, device="cuda")
    n_total = x.numel()

    result = bench_compare({
        "Triton (ours)": lambda: batchnorm1d(x, w, b),
        "PyTorch (ref)": lambda: torch.nn.functional.batch_norm(
            x, None, None, weight=w, bias=b, training=True, eps=1e-5
        ),
    }, flops=n_total * 8, bytes_accessed=n_total * 4 * 4, dtype="fp32")
    print_compare_report(result)


# PERFORMANCE NOTES
# =================
# - BatchNorm 的关键性能挑战: strided memory access
#   - 同一 channel 的连续 sample 间隔 C 个元素 (non-unit stride)
#   - L2 cache 可以缓解, 但当 C 很大时 (如 4096), cache miss 严重
# - 解决方案:
#   1. 转置输入为 (C, N) → coalesced access
#      - 代价: 一次 transpose 的 overhead
#      - cuDNN 内部就是这样做的!
#   2. 使用 shared memory tile
#      - 每个 program 加载一个 tile 到 shared memory
#      - tile 内转置后 coalesced 处理
# - 本实现是教学版 (直接 strided), 展示了 "同一个 reduction,
#   不同 memory layout → 完全不同性能" 的重要概念

if __name__ == "__main__":
    main()
