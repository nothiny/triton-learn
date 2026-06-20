"""
43_scaled_dot_product.py — Scaled Dot-Product Attention (QK^T / sqrt(d))

学习目标:
  - 掌握 Attention 的核心计算: scores = Q · K^T / sqrt(d)
  - 理解 Row-wise dot product 的 batch 实现 (2D grid)
  - 为理解 Flash Attention (Phase 2) 中的 QK^T 块打基础

数学定义:
  给定 query Q (N_heads × d) 和 key K (N_heads × d):
  scores[i] = dot(Q[i], K[i]) / sqrt(d)  for each head i

为什么叫 "scaled":
  - 除以 sqrt(d) 防止点积太大导致 softmax 饱和
  - 当 d=64, Q·K 的方差 ≈ 64 → scaling 后方差 ≈ 1
  - 没有 scaling: softmax(大值) → one-hot → 梯度消失

和 Flash Attention 的关系:
  - Flash Attention = QK^T + softmax + PV (三步)
  - 本 kernel 只做第一步 (scores), 作为 building block
  - 完整 Flash Attention 需要 tiling + online softmax (见 Phase 2)

运行: python phase1_fundamentals/43_scaled_dot_product.py
"""

import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch, triton, triton.language as tl
from utils.profiler import bench_compare, print_compare_report


@triton.jit
def scaled_dot_product_kernel(q_ptr, k_ptr, scores_ptr,
                                n_heads, d_head, scale,
                                BLOCK_HEADS: tl.constexpr):
    """
    对每个 head 计算 scaled dot product: scores[h] = Q[h]·K[h] / sqrt(d).
    每个 program 处理一批 heads.
    """
    pid = tl.program_id(0)
    head_start = pid * BLOCK_HEADS
    head_offsets = head_start + tl.arange(0, BLOCK_HEADS)
    head_mask = head_offsets < n_heads

    # Q 和 K 的布局: [n_heads, d_head], row-major
    # 每个 head 的 Q 行起始地址 = head * d_head
    q_row_starts = head_offsets * d_head
    k_row_starts = head_offsets * d_head

    # 对每个 head, 累加 Q[dim] * K[dim]
    accum = tl.zeros([BLOCK_HEADS], dtype=tl.float32)
    for d in range(0, d_head):
        q_offsets = q_row_starts + d
        k_offsets = k_row_starts + d
        d_mask = head_mask  # 只在 head 维度上 mask, d 维度保证不出界
        q_val = tl.load(q_ptr + q_offsets, mask=d_mask, other=0.0)
        k_val = tl.load(k_ptr + k_offsets, mask=d_mask, other=0.0)
        accum += q_val * k_val

    # [GPU] 除以 sqrt(d) 做 scaling
    scores = accum * scale

    tl.store(scores_ptr + head_offsets, scores, mask=head_mask)


def scaled_dot_product(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    Row-wise scaled dot product: scores[h] = Q[h]·K[h] / sqrt(d).
    输入: (n_heads, d_head) 或 (n_heads, d_head) 布局.
    """
    assert q.shape == k.shape and q.dim() == 2
    n_heads, d_head = q.shape
    scale = d_head ** -0.5
    scores = torch.empty(n_heads, device=q.device, dtype=torch.float32)
    BLOCK_HEADS = min(256, triton.next_power_of_2(n_heads))
    grid = (triton.cdiv(n_heads, BLOCK_HEADS),)
    scaled_dot_product_kernel[grid](
        q, k, scores, n_heads, d_head, scale, BLOCK_HEADS=BLOCK_HEADS,
    )
    return scores


def main():
    print("=" * 60)
    print("43_scaled_dot_product — Scaled QK^T (Attention Building Block)")
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.manual_seed(42)

    for name, (h, d) in [("tiny  ", (4, 64)), ("small ", (32, 128)),
                           ("medium", (128, 256))]:
        Q = torch.randn(h, d, device="cuda")
        K = torch.randn(h, d, device="cuda")
        scores_t = scaled_dot_product(Q, K)
        scores_r = (Q * K).sum(dim=-1) / (d ** 0.5)
        max_diff = (scores_t - scores_r).abs().max().item()
        print(f"  [{name}] heads={h} d={d}  max_diff={max_diff:.2e}  "
              f"{'✅' if max_diff < 1e-5 else '❌'}")

    # Demo: attention scores → softmax → weights
    print("\n--- Attention Score → Softmax (mini attention demo) ---")
    Q = torch.randn(8, 64, device="cuda")
    K = torch.randn(8, 64, device="cuda")
    scores = scaled_dot_product(Q, K)
    attn_weights = torch.softmax(scores, dim=-1)
    print(f"  Scaled scores (first 4): {scores[:4].tolist()}")
    print(f"  Attention weights (first 4): {attn_weights[:4].tolist()}")
    print(f"  Weights sum: {attn_weights.sum().item():.2f} (should = n_heads)")

    print("\n--- Performance ---")
    Q = torch.randn(4096, 128, device="cuda", dtype=torch.float32)
    K = torch.randn(4096, 128, device="cuda", dtype=torch.float32)
    n = Q.numel()
    result = bench_compare(
        {
            "Triton scaled dot product": lambda: scaled_dot_product(Q, K),
            "PyTorch (Q*K).sum(-1)/sqrt(d)": lambda: (Q * K).sum(dim=-1) / (128 ** 0.5),
        },
        flops=Q.shape[0] * Q.shape[1] * 2,
        bytes_accessed=n * 4 * 2,  # read Q + K
        dtype="fp32",
    )
    print_compare_report(result)


# PERFORMANCE NOTES
# =================
# - Scaled dot product = row-wise inner product + scale.
# - 每个 head 串行遍历 d_head 维 (循环), 对 small d (64-256) 影响不大.
# - 对于大 d_head (1024+), 应该用 block-level tiling (类似 matmul 的 K 维循环).
# - scale * accum: 在循环外一次完成, 不增加循环内延迟.
# - 这是 Attention 的 building block — QK^T 之后还需要 softmax 和乘以 V.

if __name__ == "__main__":
    main()
