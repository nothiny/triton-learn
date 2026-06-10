"""
22_embedding.py — Embedding Lookup / Gather kernel

学习目标：
  - 理解 embedding lookup 在 GPU 上的内存访问模式
  - 掌握非 coalesced gather 操作的实现
  - 学会在 Triton 中处理随机访存

公式:
  output[i] = weight[ids[i]]
  其中 weight 是 (vocab_size, embed_dim), ids 是 (batch_size,)

在 LLM 中的使用:
  - 每个 LLM 的第一层: token_id → embedding vector
  - vocab_size 通常 32000-128000+
  - 本质是 gather 操作: 从大表中按索引取行

性能挑战:
  - Gather 是随机访存: ids 中相邻的 token 可能对应 weight 中不相邻的行
  - 缓存命中率低 (vocab_size >> L2 cache)
  - 优化方向: 利用 shared memory cache 频繁访问的 token

运行: python phase1_fundamentals/22_embedding.py
"""

import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.profiler import bench_compare, print_compare_report


@triton.jit
def embedding_kernel(
    weight_ptr,  # (vocab_size, embed_dim) — embedding table
    ids_ptr,     # (n_tokens,) — token indices
    output_ptr,  # (n_tokens, embed_dim) — gathered embeddings
    embed_dim,
    n_tokens,
    BLOCK_SIZE: tl.constexpr,     # 沿 embed_dim 处理的元素数
):
    """
    每个 program 处理一个 token 的 embedding lookup.

    Grid = (n_tokens,)
    由于每个 token 查不同的 weight 行 → 非 coalesced 访存.
    """
    token_idx = tl.program_id(axis=0)
    token_id = tl.load(ids_ptr + token_idx)  # 该位置的 token index

    # 该 token 对应的 weight 行
    row_start = token_id * embed_dim
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # 沿 embed_dim 加载 weight 行
    for block_start in range(0, embed_dim, BLOCK_SIZE):
        offsets = row_start + block_start + col_offsets
        mask = (block_start + col_offsets) < embed_dim
        w = tl.load(weight_ptr + offsets, mask=mask, other=0.0)

        out_offsets = token_idx * embed_dim + block_start + col_offsets
        tl.store(output_ptr + out_offsets, w, mask=mask)


def embedding(weight: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """
    Embedding lookup: output[i] = weight[ids[i]].

    Args:
        weight: (vocab_size, embed_dim)
        ids: (n_tokens,) — int64 indices, 每个值在 [0, vocab_size-1]
    """
    vocab_size, embed_dim = weight.shape
    n_tokens = ids.numel()

    output = torch.empty(n_tokens, embed_dim, device=weight.device, dtype=weight.dtype)
    grid = (n_tokens,)  # 每 token 一个 program
    embedding_kernel[grid](weight, ids, output, embed_dim, n_tokens, BLOCK_SIZE=128)
    return output


def main():
    print("=" * 60)
    print("22_embedding — Embedding Lookup / Gather")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    torch.manual_seed(42)

    # ---- 正确性 ----
    vocab_size, embed_dim, n_tokens = 10000, 256, 512
    weight = torch.randn(vocab_size, embed_dim, device="cuda")
    ids = torch.randint(0, vocab_size, (n_tokens,), device="cuda")

    out_triton = embedding(weight, ids)
    out_torch = torch.nn.functional.embedding(ids, weight)

    max_diff = (out_triton - out_torch).abs().max().item()
    print(f"  vocab={vocab_size}  embed_dim={embed_dim}  "
          f"n_tokens={n_tokens}")
    print(f"  max_diff={max_diff:.2e}  "
          f"{'✅' if max_diff < 1e-5 else '❌'}")

    # ---- LLM 规模测试 ----
    vocab_size, embed_dim, n_tokens = 32000, 4096, 2048
    weight = torch.randn(vocab_size, embed_dim, device="cuda")
    ids = torch.randint(0, vocab_size, (n_tokens,), device="cuda")

    out_triton = embedding(weight, ids)
    out_torch = torch.nn.functional.embedding(ids, weight)
    max_diff = (out_triton - out_torch).abs().max().item()
    print(f"  [LLM scale] vocab={vocab_size}  embed_dim={embed_dim}  "
          f"n_tokens={n_tokens}  max_diff={max_diff:.2e}  "
          f"{'✅' if max_diff < 1e-5 else '❌'}")

    # ---- 性能对比 ----
    print("\n--- Performance ---")
    vocab_size, embed_dim, n_tokens = 50000, 1024, 4096
    weight = torch.randn(vocab_size, embed_dim, device="cuda", dtype=torch.float32)
    ids = torch.randint(0, vocab_size, (n_tokens,), device="cuda")

    total_flops = n_tokens * embed_dim  # copy = 0 FLOPs (just memory)
    total_bytes = n_tokens * embed_dim * 4 + weight.numel() * 4 // 100  # approx

    result = bench_compare({
        "Triton (ours)": lambda: embedding(weight, ids),
        "PyTorch (ref)": lambda: torch.nn.functional.embedding(ids, weight),
    }, flops=total_flops, bytes_accessed=n_tokens * embed_dim * 2 * 4, dtype="fp32")
    print_compare_report(result)


# PERFORMANCE NOTES
# =================
# - Embedding lookup 是 gather 操作 → 随机访存
#   - 连续 token 查不同 weight 行 → 地址不连续
#   - L2 cache 命中率取决于 token 分布 (Zipf 分布: 少数高频 token 大部分访问)
# - 访存模型:
#   - Best case (所有 token 相同): 1 行被反复读取 → L1/L2 100% 命中
#   - Worst case (所有 token 不同): N token × embed_dim 次独立访存 → 0% 命中
#   - 实际: Zipf 分布 → top-1000 token 占 60-80% 访问 → L2 缓存有一定效果
# - PyTorch 的 embedding 使用 CUDA kernel 直接 gather, 和我们的实现类似
# - 优化方向:
#   1. 对高频 token 使用 shared memory cache
#   2. 对 batch 中重复的 token 做 dedup → gather → expand
#   3. 使用 tensor core 做 embedding + matmul fusion (NVIDIA 的 "fused embedding")

if __name__ == "__main__":
    main()
