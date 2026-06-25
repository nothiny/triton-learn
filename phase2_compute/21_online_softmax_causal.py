"""
19_online_softmax_causal.py — Online Softmax with Causal Mask

学习目标:
  - 理解 online softmax 的数值稳定性机制
  - 掌握 causal mask 在 softmax 中的高效嵌入方式
  - 学会用 tl.where 将 mask 融入 max/sum 的计算

背景:
  Standard softmax: softmax(x_i) = exp(x_i - max) / sum(exp(x_i - max))

  Causal mask: 位置 i 只能 attend 到位置 j ≤ i
  即: score[i, j] = -inf for j > i（在 exp 前设为 0）

  Online 算法（避免存储完整矩阵）:
    for each tile of x:
      m_new = max(m_old, max(tile))
      s_new = s_old * exp(m_old - m_new) + sum(exp(tile - m_new))

  加入 causal mask:
    对于 j > i 的位置，tile 值设为 -inf (exp → 0)

运行: python phase2_compute/19_online_softmax_causal.py
"""

import torch
import triton
import triton.language as tl
from triton.testing import do_bench


@triton.jit
def online_softmax_causal_kernel(
    x_ptr, output_ptr,
    N,                  # sequence length
    stride_xn,
    BLOCK_N: tl.constexpr,
):
    """
    Per-row online softmax with causal mask.

    每个 program 处理一行，沿 N 维 tile 迭代。

    causal mask: score[i, j] = -inf if j > i
    即第 i 行只对 j <= i 的位置有效。
    """
    row_idx = tl.program_id(axis=0)

    # Online softmax 状态
    m = tl.full([1], float("-inf"), dtype=tl.float32)
    s = tl.zeros([1], dtype=tl.float32)

    # Pass 1: 计算 running max m 和 running sum s
    num_tiles = tl.cdiv(N, BLOCK_N)
    for tile_idx in range(num_tiles):
        col_offs = tile_idx * BLOCK_N + tl.arange(0, BLOCK_N)

        # Load tile
        x_ptrs = x_ptr + row_idx * stride_xn + col_offs
        tile = tl.load(x_ptrs, mask=col_offs < N, other=float("-inf"))

        # 应用 causal mask: j > i → -inf
        causal_mask = col_offs <= row_idx
        tile = tl.where(causal_mask, tile, float("-inf"))

        # Online softmax update
        tile_m = tl.max(tile, axis=0)
        m_new = tl.maximum(m, tile_m)
        s = s * tl.exp(m - m_new) + tl.sum(tl.exp(tile - m_new), axis=0)
        m = m_new

    # Pass 2: softmax(x) = exp(x - m) / s，并写回
    for tile_idx in range(num_tiles):
        col_offs = tile_idx * BLOCK_N + tl.arange(0, BLOCK_N)

        x_ptrs = x_ptr + row_idx * stride_xn + col_offs
        tile = tl.load(x_ptrs, mask=col_offs < N, other=float("-inf"))

        causal_mask = col_offs <= row_idx
        tile = tl.where(causal_mask, tile, float("-inf"))

        out_tile = tl.exp(tile - m) / s
        out_ptrs = output_ptr + row_idx * stride_xn + col_offs
        tl.store(out_ptrs, out_tile, mask=col_offs < N)


def online_softmax_causal(x: torch.Tensor) -> torch.Tensor:
    """Row-wise causal softmax: (M, N) → (M, N) with lower-triangular mask."""
    M, N = x.shape
    output = torch.empty_like(x)

    grid = (M,)
    online_softmax_causal_kernel[grid](
        x, output, N, x.stride(0), BLOCK_N=128,
    )
    return output


def ref_causal_softmax(x: torch.Tensor) -> torch.Tensor:
    """PyTorch reference: causal softmax"""
    M, N = x.shape
    row_idx = torch.arange(M, device=x.device)[:, None]
    col_idx = torch.arange(N, device=x.device)[None, :]
    causal_mask = (col_idx <= row_idx)
    x_masked = x.float().masked_fill(~causal_mask, float("-inf"))
    return torch.softmax(x_masked, dim=-1).to(x.dtype)


def main():
    print("=" * 60)
    print("19_online_softmax_causal — Online Softmax with Causal Mask")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    configs = [
        (64, 64),
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 1024),
    ]

    for M, N in configs:
        x = torch.randn(M, N, device="cuda", dtype=torch.float32)

        out_triton = online_softmax_causal(x)
        out_ref = ref_causal_softmax(x)

        max_diff = (out_triton - out_ref).abs().max().item()

        ms = do_bench(lambda: online_softmax_causal(x))
        bandwidth = (x.numel() * 2 * 4) / (ms * 1e-3) / 1e9  # read + write, fp32

        status = "✅" if max_diff < 1e-4 else "❌"
        print(f"  {M}×{N}: {ms:.4f}ms  {bandwidth:.1f} GB/s  "
              f"diff={max_diff:.2e}  {status}")


# PERFORMANCE NOTES
# =================
# - Online softmax 是 Flash Attention 的核心组件:
#   不需要完整存储 attention matrix，只需要 O(N) 而不是 O(N²) 内存
# - Causal mask 通过 tl.where 在线应用: 被 mask 的位置设为 -inf → exp(-inf) = 0
# - 2-pass 实现 (先算 m/s，再写回):
#   - Pass 1: 计算 running max m 和 running sum s
#   - Pass 2: 用 m 和 s 做归一化并写回
# - Memory-bound: 2-pass 意味着读 2 次 + 写 1 次 = 3x N² 字节
# - 可以与 attention matmul 融合: 在累加 exp(score) 时同时累加 softmax 的分母


if __name__ == "__main__":
    main()
