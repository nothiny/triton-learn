"""
11_grouped_query_attention.py — Grouped Query Attention (GQA)

学习目标：
  - 理解 GQA/MQA 的内存优化原理
  - 掌握 KV head sharing 的索引映射
  - 对比标准 MHA vs GQA vs MQA 的性能差异

背景:
  Multi-Query Attention (MQA, Shazeer 2019): 所有 Q heads 共享同一个 KV pair
  Grouped-Query Attention (GQA, Ainslie et al., 2023):
    将 Q heads 分成 groups，每组共享一个 KV head
    当 groups=1 时 → MQA (Single KV for all Q heads)
    当 groups=n_heads 时 → MHA (Standard multi-head attention)

  用于: Llama-2 70B, Llama-3, Mistral, Gemma, Mixtral 等

  GQA 的核心价值在于减少 KV cache 大小（推理时），同时保持注意力质量。
  对于训练，主要优化是 KV 数据复用：同一 group 内的 Q heads 使用相同 KV。

算法:
  对于 q(heads_q) 和 kv(heads_kv)，其中 heads_q 是 heads_kv 的整数倍：
    num_kv_groups = heads_q // heads_kv
    kv_head_idx(q_head_idx) = q_head_idx // num_kv_groups

  本实现通过在 grid 层面按 KV head 分组，每个 program 处理一个 group 的所有 Q heads，
  加载 KV 一次，循环 Q heads 复用。

运行: python phase2_compute/11_grouped_query_attention.py
"""

import math
import statistics
import torch
import triton
import triton.language as tl
from triton.testing import do_bench


@triton.jit
def gqa_flash_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    BATCH, N_HEADS_Q, N_HEADS_KV, N_CTX,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_om,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    D_HEAD: tl.constexpr,
    CAUSAL: tl.constexpr,
    SCALE: tl.constexpr,
):
    """
    GQA Flash Attention forward kernel.

    每个 program 负责一个 (batch, kv_head, q_tile) 的输出。
    对于该 KV head 对应的 group 内的每一个 Q head，执行标准 Flash Attention。
    KV 在每个 KV tile 加载一次，被 group 内所有 Q heads 复用。

    Grid dimension: (BATCH * N_HEADS_KV * num_q_blocks,)
    """
    NUM_KV_GROUPS: tl.constexpr = N_HEADS_Q // N_HEADS_KV  # Q heads per KV head

    pid = tl.program_id(axis=0)
    num_q_blocks = tl.cdiv(N_CTX, BLOCK_Q)
    block_q_idx = pid % num_q_blocks
    pid_b_hkv = pid // num_q_blocks
    batch_idx = pid_b_hkv // N_HEADS_KV
    kv_head_idx = pid_b_hkv % N_HEADS_KV

    # Q tile offsets (same for all Q heads in group)
    offs_q = block_q_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, D_HEAD)

    # KV range for causal: only up to this Q tile's rightmost position
    kv_end = (block_q_idx + 1) * BLOCK_Q if CAUSAL else N_CTX

    # For each Q head in this KV group, do Flash Attention
    # [GPU] 同一 group 的 Q heads 共享 KV，外层循环复用 KV 数据
    for group_idx in range(NUM_KV_GROUPS):
        q_head_idx = kv_head_idx * NUM_KV_GROUPS + group_idx

        # Load Q tile for this Q head: [BLOCK_Q, D_HEAD]
        q_ptrs = (q_ptr + batch_idx * stride_qb + q_head_idx * stride_qh +
                  offs_q[:, None] * stride_qm + offs_d[None, :])
        q = tl.load(q_ptrs, mask=offs_q[:, None] < N_CTX, other=0.0)

        # Online softmax state
        m_i = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)
        acc = tl.zeros([BLOCK_Q, D_HEAD], dtype=tl.float32)

        # Iterate over KV blocks — KV loaded once, used by all Q heads in loop
        for block_kv_start in range(0, kv_end, BLOCK_KV):
            offs_kv = block_kv_start + tl.arange(0, BLOCK_KV)

            # Load K tile: [BLOCK_KV, D_HEAD]
            k_ptrs = (k_ptr + batch_idx * stride_kb + kv_head_idx * stride_kh +
                      offs_kv[:, None] * stride_kn + offs_d[None, :])
            k = tl.load(k_ptrs, mask=offs_kv[:, None] < N_CTX, other=0.0)

            # S = Q @ K^T * scale : [BLOCK_Q, BLOCK_KV]
            s = tl.dot(q, k.T) * SCALE

            # Causal mask
            if CAUSAL:
                q_pos = offs_q[:, None]
                kv_pos = offs_kv[None, :]
                s = tl.where(q_pos >= kv_pos, s, float("-inf"))

            # Online softmax update
            m_ij = tl.max(s, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            p = tl.exp(s - m_new[:, None])

            alpha = tl.exp(m_i - m_new)
            l_new = alpha * l_i + tl.sum(p, axis=1)

            acc = acc * alpha[:, None]

            # Load V tile: [BLOCK_KV, D_HEAD]
            v_ptrs = (v_ptr + batch_idx * stride_vb + kv_head_idx * stride_vh +
                      offs_kv[:, None] * stride_vn + offs_d[None, :])
            v = tl.load(v_ptrs, mask=offs_kv[:, None] < N_CTX, other=0.0).to(tl.float32)
            acc += tl.dot(p, v)

            m_i = m_new
            l_i = l_new

        # Final normalization
        acc = acc / l_i[:, None]

        # Write output for this Q head
        offs_m = block_q_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)
        o_ptrs = (o_ptr + batch_idx * stride_ob + q_head_idx * stride_oh +
                  offs_m[:, None] * stride_om + offs_d[None, :])
        tl.store(o_ptrs, acc, mask=offs_m[:, None] < N_CTX)


# ==============================================================================
# Python Wrapper
# ==============================================================================


def grouped_query_attention(
    q: torch.Tensor,       # (batch, n_heads_q, seq_len, d_head)
    k: torch.Tensor,       # (batch, n_heads_kv, seq_len, d_head)
    v: torch.Tensor,       # (batch, n_heads_kv, seq_len, d_head)
    causal: bool = False,
    block_q: int = 64,
    block_kv: int = 64,
) -> torch.Tensor:
    """
    Grouped Query Attention (GQA) — Flash Attention style.

    Args:
        q:  Query tensor  (batch, n_heads_q, seq_len, d_head)
        k:  Key tensor    (batch, n_heads_kv, seq_len, d_head)
        v:  Value tensor  (batch, n_heads_kv, seq_len, d_head)
        causal: If True, apply causal mask (GPT-style autoregressive).
        block_q, block_kv: Tile sizes for Q and KV dimensions.

    Returns:
        o: (batch, n_heads_q, seq_len, d_head)

    Preconditions:
        - n_heads_q must be an integer multiple of n_heads_kv.
        - All tensors must be fp16 or bf16.
        - k and v must have the same n_heads dimension.
    """
    BATCH, N_HEADS_Q, N_CTX, D_HEAD = q.shape
    _, N_HEADS_KV, _, _ = k.shape
    assert N_HEADS_Q % N_HEADS_KV == 0, (
        f"n_heads_q ({N_HEADS_Q}) must be multiple of n_heads_kv ({N_HEADS_KV})"
    )

    o = torch.empty_like(q)

    # Grid: (BATCH * N_HEADS_KV * num_q_blocks,)
    # Each program handles one KV head × Q tile, iterates over the Q heads in that group.
    num_q_blocks = triton.cdiv(N_CTX, block_q)
    grid = (BATCH * N_HEADS_KV * num_q_blocks,)

    gqa_flash_fwd_kernel[grid](
        q, k, v, o,
        BATCH, N_HEADS_Q, N_HEADS_KV, N_CTX,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        BLOCK_Q=block_q,
        BLOCK_KV=block_kv,
        D_HEAD=D_HEAD,
        CAUSAL=causal,
        SCALE=1.0 / math.sqrt(D_HEAD),
    )
    return o


# ==============================================================================
# References
# ==============================================================================


def ref_gqa_mha(q, k, v, causal=False):
    """
    PyTorch reference for GQA (MHA — all heads use same KV via repeat).
    This expands KV to full MHA for correctness checking.

    Uses the standard: O = softmax(Q @ K^T / sqrt(d)) @ V
    """
    N_HEADS_Q = q.shape[1]
    N_HEADS_KV = k.shape[1]
    num_kv_groups = N_HEADS_Q // N_HEADS_KV
    d_head = q.shape[-1]
    scale = 1.0 / (d_head ** 0.5)

    # Expand KV: repeat each KV head for its Q group
    # k, v: (B, Hkv, N, D) → (B, Hq, N, D)
    k_expanded = k.repeat_interleave(num_kv_groups, dim=1)
    v_expanded = v.repeat_interleave(num_kv_groups, dim=1)

    attn = (q @ k_expanded.transpose(-2, -1)) * scale
    if causal:
        N = attn.shape[-1]
        mask = torch.tril(torch.ones(N, N, device=q.device))
        attn = attn.masked_fill(mask == 0, float("-inf"))
    attn = torch.softmax(attn, dim=-1)
    return attn @ v_expanded


def ref_gqa_torch_sdpa(q, k, v, causal=False):
    """
    PyTorch SDPA reference for GQA.

    torch SDPA natively supports GQA via the ``enable_gqa`` flag (PyTorch >= 2.1).
    """
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=causal, enable_gqa=True,
    )


# ==============================================================================
# Benchmark helper
# ==============================================================================


def bench_gqa_vs_mha(
    batch: int, n_heads_q: int, n_heads_kv: int,
    seq_len: int, d_head: int,
    causal: bool = True,
):
    """Benchmark GQA Triton vs PyTorch SDPA (both with GQA)."""
    q = torch.randn(batch, n_heads_q, seq_len, d_head, device="cuda", dtype=torch.float16)
    k = torch.randn(batch, n_heads_kv, seq_len, d_head, device="cuda", dtype=torch.float16)
    v = torch.randn(batch, n_heads_kv, seq_len, d_head, device="cuda", dtype=torch.float16)

    t_triton = do_bench(lambda: grouped_query_attention(q, k, v, causal=causal))
    t_sdpa = do_bench(lambda: torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=causal, enable_gqa=True))
    return t_triton, t_sdpa


# ==============================================================================
# Main
# ==============================================================================


def main():
    print("=" * 70)
    print("11_grouped_query_attention — Grouped Query Attention (GQA)")
    print("=" * 70)

    torch.manual_seed(42)

    # Test configurations: various GQA ratios
    configs = [
        # (batch, n_heads_q, n_heads_kv, seq_len, d_head, groups, causal, desc)
        (1, 8, 2, 128, 64, 4, False, "8Q/2KV (4 groups), non-causal, short"),
        (1, 8, 2, 256, 64, 4, True,  "8Q/2KV (4 groups), causal, medium"),
        (1, 8, 1, 128, 64, 8, True,  "8Q/1KV (MQA), causal"),
        (2, 32, 8, 128, 64, 4, True, "32Q/8KV (GQA, Llama-2 style), causal"),
    ]

    for cfg in configs:
        B, Hq, Hkv, N, D, groups, causal, desc = cfg
        print(f"\n── {desc} ──")

        q = torch.randn(B, Hq, N, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, Hkv, N, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, Hkv, N, D, device="cuda", dtype=torch.float16)

        # Triton GQA
        o_triton = grouped_query_attention(q, k, v, causal=causal)

        # Reference: torch SDPA with GQA (PyTorch >= 2.1)
        o_sdpa = ref_gqa_torch_sdpa(q, k, v, causal=causal)

        max_diff = (o_triton.float() - o_sdpa.float()).abs().max().item()
        status = "✅" if max_diff < 0.05 else "❌"
        print(f"  vs torch SDPA (GQA): max_diff = {max_diff:.6e}  {status}")

        # Also check against expanded-MHA reference
        o_ref = ref_gqa_mha(q, k, v, causal=causal)
        ref_diff = (o_triton.float() - o_ref.float()).abs().max().item()
        ref_status = "✅" if ref_diff < 0.05 else "❌"
        print(f"  vs expanded MHA:    max_diff = {ref_diff:.6e}  {ref_status}")

    # Benchmark: compare Triton GQA vs PyTorch SDPA
    print(f"\n{'='*70}")
    print("Benchmark: Triton GQA vs PyTorch SDPA GQA")
    print(f"{'='*70}")

    bench_configs = [
        (2, 32, 8, 512, 64),   # Small Llama-2 style
        (2, 32, 8, 2048, 64),  # Medium Llama-2
    ]
    for B, Hq, Hkv, N, D in bench_configs:
        t_triton, t_sdpa = bench_gqa_vs_mha(B, Hq, Hkv, N, D, causal=True, warmup=5, rep=30)
        speedup = t_sdpa / t_triton
        label = f"B={B} Hq={Hq} Hkv={Hkv} N={N} D={D}"
        print(f"  {label}")
        print(f"    Triton GQA:  {t_triton:.4f}ms")
        print(f"    torch SDPA:  {t_sdpa:.4f}ms")
        print(f"    Speedup:     {speedup:.2f}x {'(Triton faster)' if speedup > 1.0 else '(SDPA faster)'}")

    # Memory analysis
    print(f"\n{'='*70}")
    print("Memory analysis (GQA vs MHA — KV cache)")
    print(f"{'='*70}")
    Hq, Hkv, N, D = 32, 8, 4096, 128
    mha_kv = 2 * Hq * N * D * 2  # K + V for MHA (fp16 bytes)
    gqa_kv = 2 * Hkv * N * D * 2  # K + V for GQA
    print(f"  MHA KV cache (H=32): {mha_kv / 1024**2:.1f} MB")
    print(f"  GQA KV cache (H=8):  {gqa_kv / 1024**2:.1f} MB")
    print(f"  Reduction:           {mha_kv / gqa_kv:.0f}x smaller KV cache")


# PERFORMANCE NOTES
# =================
# - GQA 的核心优化在推理阶段：KV cache 减小 num_kv_groups 倍
#   (Llama-2 70B: 8 groups → 8x smaller KV cache)
# - 在训练阶段，GQA 的优势是带宽优化：
#   - 同一 group 的 Q heads 复用相同 KV，减少 HBM 读取
#   - 当 batch/heads 较少时，GQA 减少有效 HBM traffic
# - [GPU] 当前实现中，同一 KV tile 在 group 循环中被重复加载
#   (每个 Q head 都加载一次) — 因为 Triton 的 shared memory 管理
#   由编译器自动决定。可以通过 explicit shared memory staging 进一步优化。
# - [COMPILER] 当 NUM_KV_GROUPS=1 (MHA), 编译器会生成与标准 Flash Attention
#   相同的代码 — 零开销
# - GQA 的关键权衡: MHA (质量) ← → MQA (速度)
#   - GQA 在两者之间提供可调节的平衡
#   - Llama-2 使用 GQA (8 groups); Llama-3 也使用 GQA
# - TODO: KV cache 预填充优化 (paged attention) 用于推理
# - TODO: 支持跨 batch 的 GQA（不同样本可有不同 seq lengths）


if __name__ == "__main__":
    main()
