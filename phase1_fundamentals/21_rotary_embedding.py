"""
12_rotary_embedding.py — Rotary Position Embedding (RoPE) kernel

学习目标：
  - 理解 RoPE 在 LLM 中的位置编码机制（Llama, Mistral, Qwen 都用）
  - 掌握 Triton 中 pairwise 操作的实现模式
  - 学习 sin/cos 预计算 + kernel 应用的分离设计

数学公式:
  对于位置 pos 和维度对 (2i, 2i+1):
    θ_i = base^(-2i/d)  where base = 10000 (或 500000 for Llama 3)
    cos_i = cos(pos * θ_i), sin_i = sin(pos * θ_i)

    x'[2i]   = x[2i] * cos_i - x[2i+1] * sin_i
    x'[2i+1] = x[2i+1] * cos_i + x[2i] * sin_i

  等价向量形式:
    x' = x * cos + rotate_half(x) * sin
    其中 rotate_half([a, b, c, d, ...]) = [-b, a, -d, c, ...]

RoPE 的特性:
  1. 相对位置编码: q_i · k_j 只依赖于相对位置 (i-j)
  2. 远程衰减: 点积随距离指数衰减 → 自然的局部性
  3. 绝对 + 相对: 既编码绝对位置, 又保持相对位置关系

在 LLM 中的使用:
  - 每次 attention 前对 Q 和 K 应用 RoPE (不对 V 应用)
  - head_dim 通常在 64-128 之间, 即 32-64 个旋转对
  - 现代 LLM 使用更大的 base (如 500000) 来支持更长的上下文

运行: python phase1_fundamentals/21_rotary_embedding.py
"""

import math
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.profiler import bench_compare, print_compare_report


# ---------------------------------------------------------------------------
# RoPE 预计算 (CPU 端)
# ---------------------------------------------------------------------------


def precompute_freqs_cis(head_dim: int, seq_len: int,
                         base: float = 10000.0,
                         device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """
    预计算 RoPE 的 cos 和 sin 值。

    Args:
        head_dim: 每个 attention head 的维度 (必须是偶数)
        seq_len: 最大序列长度
        base: RoPE base frequency (Llama 1/2: 10000, Llama 3: 500000)
        device: 目标设备

    Returns:
        cos: (seq_len, head_dim) — cosine 值
        sin: (seq_len, head_dim) — sine 值
    """
    assert head_dim % 2 == 0, "head_dim must be even"
    half_dim = head_dim // 2

    # θ_i = base^(-2i/d), i = 0, 1, ..., half_dim-1
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))

    # 位置: [0, 1, ..., seq_len-1]
    positions = torch.arange(seq_len, dtype=torch.float32)

    # 外积: (seq_len, half_dim) = (seq_len, 1) × (half_dim,)
    # freq[pos, i] = pos * θ_i
    freqs = torch.outer(positions, theta)  # (seq_len, half_dim)

    # 扩展回 head_dim: 每个 θ 对应一对维度
    # [cos_0, cos_0, cos_1, cos_1, ...] — 成对重复
    cos = freqs.cos().repeat_interleave(2, dim=-1).to(device)
    sin = freqs.sin().repeat_interleave(2, dim=-1).to(device)

    return cos, sin


# ---------------------------------------------------------------------------
# PyTorch reference
# ---------------------------------------------------------------------------


def rope_pytorch(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    PyTorch 参考实现: x' = x * cos + rotate_half(x) * sin.

    Args:
        x: (batch, n_heads, seq_len, head_dim) 或 (seq_len, head_dim)
        cos: 广播兼容的 cosine, shape (seq_len, head_dim)
        sin: 广播兼容的 sine, shape (seq_len, head_dim)

    Returns:
        旋转后的 x
    """
    # rotate_half: 交换每对中的两个元素并取负第一个
    x_rotated = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1)
    x_rotated = x_rotated.flatten(-2)  # 还原 shape

    return x * cos + x_rotated * sin


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------


@triton.jit
def rope_kernel(
    x_ptr,       # 输入: 展平为 (total_elements,)
    cos_ptr,     # cos 值: 和 x 同 shape (已广播)
    sin_ptr,     # sin 值: 和 x 同 shape (已广播)
    output_ptr,  # 输出: 和 x 同 shape
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    RoPE kernel: output[i] = x[i] * cos[i] + rotate_half(x)[i] * sin[i]

    由于 RoPE 操作在相邻元素对上, 每对 (2j, 2j+1) 一起处理。
    每个线程处理一对元素 (2 个连续元素)。
    """
    pid = tl.program_id(axis=0)

    # 每个 program 处理 BLOCK_SIZE 对 = BLOCK_SIZE*2 个元素
    pair_start = pid * BLOCK_SIZE  # pair index
    pair_offsets = pair_start + tl.arange(0, BLOCK_SIZE)

    # 每个 pair 在原始张量中的偏移
    elem_offsets_0 = pair_offsets * 2      # 偶数位置: 2j
    elem_offsets_1 = pair_offsets * 2 + 1  # 奇数位置: 2j+1

    pair_mask = pair_offsets * 2 < n_elements

    # 加载一对元素 [x_2j, x_2j+1]
    x0 = tl.load(x_ptr + elem_offsets_0, mask=pair_mask, other=0.0)
    x1 = tl.load(x_ptr + elem_offsets_1, mask=pair_mask, other=0.0)

    # 加载对应的 cos 和 sin (cos/sin 在每对内相同)
    c = tl.load(cos_ptr + elem_offsets_0, mask=pair_mask, other=1.0)
    s = tl.load(sin_ptr + elem_offsets_0, mask=pair_mask, other=0.0)

    # ---- 2D 旋转: [x0']   = [cos  -sin] [x0] ----
    #              [x1']     [sin   cos] [x1]
    # x0' = x0 * cos - x1 * sin
    # x1' = x1 * cos + x0 * sin
    new_x0 = x0 * c - x1 * s
    new_x1 = x1 * c + x0 * s

    # 写回
    tl.store(output_ptr + elem_offsets_0, new_x0, mask=pair_mask)
    tl.store(output_ptr + elem_offsets_1, new_x1, mask=pair_mask)


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    对输入张量应用 Rotary Position Embedding。

    Args:
        x: 输入 (batch, n_heads, seq_len, head_dim) 或 (seq_len, head_dim)
        cos: (seq_len, head_dim) — 会自动广播
        sin: (seq_len, head_dim) — 会自动广播

    Returns:
        旋转后的张量, shape 与 x 相同
    """
    # 广播 cos/sin 到 x 的 shape
    orig_shape = x.shape
    while cos.dim() < x.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    # 展平所有维度
    x_flat = x.reshape(-1)
    cos_flat = cos.expand_as(x).reshape(-1)
    sin_flat = sin.expand_as(x).reshape(-1)

    output_flat = torch.empty_like(x_flat)
    n_elements = x_flat.numel()
    n_pairs = n_elements // 2

    # grid: 每个 program 处理 BLOCK_SIZE 对
    grid = lambda meta: (triton.cdiv(n_pairs, meta["BLOCK_SIZE"]),)
    rope_kernel[grid](x_flat, cos_flat, sin_flat, output_flat, n_elements, BLOCK_SIZE=512)

    return output_flat.reshape(orig_shape)


def main():
    print("=" * 60)
    print("12_rotary_embedding — Rotary Position Embedding (RoPE)")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # ---- 正确性测试 ----
    torch.manual_seed(42)

    head_dim = 128
    seq_len = 2048
    base = 10000.0

    # 预计算 cos/sin
    cos, sin = precompute_freqs_cis(head_dim, seq_len, base=base, device="cuda")

    test_cases = [
        ("2D-input", torch.randn(seq_len, head_dim, device="cuda")),
        ("4D-small", torch.randn(2, 8, seq_len, head_dim, device="cuda")),
        ("4D-large", torch.randn(4, 32, seq_len, head_dim, device="cuda")),
    ]

    for name, x in test_cases:
        out_triton = apply_rotary_emb(x, cos, sin)
        out_torch = rope_pytorch(x, cos, sin)

        max_diff = (out_triton - out_torch).abs().max().item()
        status = "✅" if max_diff < 1e-4 else "❌"
        print(f"  [{name}] shape={list(x.shape)}  "
              f"max_diff={max_diff:.2e}  {status}")

    # ---- 数值特性验证 ----
    # RoPE 不改变向量的范数 (旋转保持 L2 范数)
    x = torch.randn(4, 8, 16, head_dim, device="cuda")
    cos_small, sin_small = precompute_freqs_cis(head_dim, 16, base=base, device="cuda")
    out = apply_rotary_emb(x, cos_small, sin_small)

    norm_before = x.norm(dim=-1)
    norm_after = out.norm(dim=-1)
    norm_diff = (norm_before - norm_after).abs().max().item()
    print(f"  [norm-preserve] max L2 norm change: {norm_diff:.6e}  "
          f"{'✅' if norm_diff < 1e-4 else '❌'}")

    # ---- 性能对比: Triton vs PyTorch ----
    print("\n--- Performance ---")

    x = torch.randn(4, 32, 2048, 128, device="cuda", dtype=torch.float32)
    cos, sin = precompute_freqs_cis(128, 2048, base=base, device="cuda")

    n_elements = x.numel()
    # 每对: 4 mul + 2 add/sub = 6 FLOPs → 3 FLOPs per element
    flops_total = n_elements * 3
    # x read (4B) + cos read (4B) + sin read (4B) + out write (4B)
    bytes_total = n_elements * 4 * 4

    result = bench_compare(
        {
            "Triton (ours)": lambda: apply_rotary_emb(x, cos, sin),
            "PyTorch (ref)": lambda: rope_pytorch(x, cos, sin),
        },
        flops=flops_total,
        bytes_accessed=bytes_total,
        dtype="fp32",
    )
    print_compare_report(result)


# PERFORMANCE NOTES
# =================
# - RoPE 是 memory-bound: 3 FLOPs / 16 bytes = 0.19 FLOP/byte
#   (读 x + cos + sin, 写 output → 4 次 HBM 访问)
# - rotate_half 不需要显式计算: 在 pair-based kernel 中直接交换
# - 为什么只对 Q 和 K 应用 RoPE (不对 V):
#   - 位置信息只需要在 attention score 中体现 (Q·K^T)
#   - V 只是 value 的加权聚合, 不需要位置编码
# - 不同模型的 base 选择:
#   - Llama 1/2: base=10000
#   - Llama 3/3.1: base=500000 (更好的长上下文支持)
#   - 更大的 base → θ_i 更小 → 高频旋转更慢 → 更好的远程关系
# - 和 PyTorch 性能对比:
#   - PyTorch 使用向量化的 sin/cos + multiply + rotate_half
#   - Triton 的 pair-based kernel 合并了读写 (每对一起加载)
#   - 在长序列上, 两者的 bottleneck 都是 HBM 带宽
# - 进一步优化:
#   - Fused Q/K projection + RoPE (减少 HBM round-trip)
#   - 对于大 batch, 可以用 tensor core 加速旋转 (2x2 矩阵乘法)

if __name__ == "__main__":
    main()
