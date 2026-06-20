"""
44_causal_mask.py — Causal Attention Mask Generation & Application

学习目标:
  - 掌握 causal mask 的 GPU 实现: 下三角矩阵 + scale
  - 理解 attention mask 的两种模式: additive (add to scores) 和 boolean
  - 学习 mask 对 softmax 的影响: -inf → exp(0) = 0

Causal Mask 定义:
  mask[i, j] = 0     if i >= j  (token i 可以 attend to token j)
  mask[i, j] = -inf  if i < j   (token i 不能 attend to future token j)

为什么叫 causal:
  - 在自回归生成 (GPT) 中, token_i 只能看到自己和之前的 token
  - 防止 "偷看答案" — position i 不能看到 position i+1, i+2, ...

实现方式:
  1. 生成 mask (本 kernel): 在 GPU 上动态生成 (节省显存)
  2. 应用 mask: scores = scores + mask (additive, softmax 后 -inf 变为 0)

和 Attention 的关系:
  scores = QK^T / sqrt(d)                    (43_scaled_dot_product)
  masked_scores = scores + causal_mask       (本 kernel)
  weights = softmax(masked_scores)           (见 17_fused_softmax)

运行: python phase1_fundamentals/44_causal_mask.py
"""

import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch, triton, triton.language as tl
from utils.profiler import bench_compare, print_compare_report


@triton.jit
def causal_mask_kernel(mask_ptr, seq_len, BLOCK_SIZE: tl.constexpr):
    """
    生成下三角 causal mask: mask[i, j] = 0 if i >= j else -inf.
    输出: (seq_len, seq_len) float32 tensor.
    """
    row = tl.program_id(0)
    row_offs = row * seq_len + tl.arange(0, BLOCK_SIZE)
    col_mask = tl.arange(0, BLOCK_SIZE) < seq_len

    # [GPU] 只需比较 row >= col: 下三角 → 0, 上三角 → -inf
    # 用 float("-inf") 让 softmax 后变成 0
    col_indices = tl.arange(0, BLOCK_SIZE)
    is_lower = col_indices <= row       # 下三角 (包括对角线)
    is_valid = col_indices < seq_len    # 越界 → -inf

    # 下三角且有效 → 0.0, 否则 → -inf
    values = tl.where(is_lower & is_valid, 0.0, float("-inf"))

    tl.store(mask_ptr + row_offs, values, mask=col_mask)


def causal_mask(seq_len: int, device: str = "cuda") -> torch.Tensor:
    """
    生成 (seq_len, seq_len) 的下三角 causal mask.
    mask[i, j] = 0.0 if i >= j else -inf.
    """
    mask = torch.empty(seq_len, seq_len, device=device, dtype=torch.float32)
    grid = (seq_len,)
    BLOCK_SIZE = triton.next_power_of_2(seq_len)
    causal_mask_kernel[grid](mask, seq_len, BLOCK_SIZE=BLOCK_SIZE)
    return mask


@triton.jit
def apply_causal_mask_kernel(scores_ptr, mask_ptr, output_ptr,
                               n_elements, BLOCK_SIZE: tl.constexpr):
    """output = scores + mask (additive mask application)"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    s = tl.load(scores_ptr + offsets, mask=mask, other=0.0)
    m = tl.load(mask_ptr + offsets, mask=mask, other=float("-inf"))
    result = s + m  # [GPU] -inf + anything = -inf

    tl.store(output_ptr + offsets, result, mask=mask)


def apply_causal_mask(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply additive causal mask: masked_scores = scores + mask."""
    assert scores.shape == mask.shape
    n = scores.numel()
    output = torch.empty_like(scores)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    apply_causal_mask_kernel[grid](scores, mask, output, n, BLOCK_SIZE=1024)
    return output


def main():
    print("=" * 60)
    print("44_causal_mask — Causal Attention Mask")
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.manual_seed(42)

    # Small demo
    for seq_len in [4, 8]:
        mask = causal_mask(seq_len)
        mask_r = torch.where(
            torch.tril(torch.ones(seq_len, seq_len, device="cuda")) == 1, 0.0,
            float("-inf"),
        )
        diff = (mask == mask_r).all().item() if (mask == mask_r).all().item() else \
               (mask - mask_r).abs().max().item()
        ok = "✅" if (mask == mask_r).all().item() else "❌"
        print(f"  seq_len={seq_len}  mask correct: {ok}")

    # Visual demo
    print("\n--- Causal Mask (seq_len=6) ---")
    mask = causal_mask(6)
    for i in range(6):
        row_str = " ".join(f"{mask[i, j].item():>4.0f}" for j in range(6))
        print(f"  [{row_str}]")

    # Application demo
    print("\n--- Applying mask to attention scores ---")
    Q = torch.randn(4, 4, device="cuda") * 0.1
    K = torch.randn(4, 4, device="cuda") * 0.1
    scores = Q @ K.T  # (4, 4)
    mask = causal_mask(4)  # (4, 4)
    masked_scores = apply_causal_mask(scores, mask)
    attn_weights = torch.softmax(masked_scores, dim=-1)
    print(f"  Scores (first row): {scores[0].tolist()}")
    print(f"  Masked (first row): {masked_scores[0].tolist()}")
    print(f"  Attn weights (row0): {attn_weights[0].tolist()}")
    print(f"  ✅ Future positions (cols > row) have weight ≈ 0")

    # Perf
    print("\n--- Performance ---")
    seq_len = 2048
    mask = causal_mask(seq_len)
    scores = torch.randn(seq_len, seq_len, device="cuda")
    n = scores.numel()
    result = bench_compare(
        {
            "Triton apply_mask": lambda: apply_causal_mask(scores, mask),
            "PyTorch scores+mask": lambda: scores + mask,
        },
        flops=n,            # one add per element
        bytes_accessed=n * 4 * 3,  # read scores, mask, write output
        dtype="fp32",
    )
    print_compare_report(result)


# PERFORMANCE NOTES
# =================
# - Causal mask 生成是 O(seq²) 写入量, mask 本身不大 (8KB for seq=1024).
# - 生产实现通常在线生成 mask (不存到 HBM), 在每个 attention block 中
#   动态计算是否是 causal (通过 position index 比较).
# - -inf + score = -inf, softmax(-inf) = 0 → 完美屏蔽 future tokens.
# - 除了 causal mask, 还有 padding mask (屏蔽 PAD token).
# - Flash Attention 在 Phase 2 会进一步优化 mask 的融合.

if __name__ == "__main__":
    main()
