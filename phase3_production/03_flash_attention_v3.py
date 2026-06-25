"""
phase3_production/03_flash_attention_v3.py — Production FlashAttention (Block Ptr, GQA, Causal)

参考: flash-linear-attention (fla-org) 的 naive_attn_decoding_kernel 生产级实现

学习目标:
  - tl.make_block_ptr 处理 Q/K/V/O 全部张量
  - GQA (Grouped Query Attention) 的 striding 和 head mapping
  - Causal mask 的正确应用
  - Online softmax 的 rescaling 技巧
  - V 维 tiling (多 program 并行处理 V 维度)

对比:
  phase2_compute/12_flash_attention_v1.py — 手工指针拼接的老写法
  phase2_compute/13_flash_attention_v2.py — causal 版本但也是手工指针

运行: python phase3_production/03_flash_attention_v3.py
"""

import torch
import triton
import triton.language as tl
from triton.testing import do_bench


@triton.autotune(
    configs=[
        triton.Config({"BQ": bq, "BV": bv}, num_warps=w, num_stages=s)
        for bq in [32, 64] for bv in [64, 128]
        for w in [4, 8] for s in [2, 3]
    ],
    key=["N_CTX", "D_HEAD"],
)
@triton.jit
def flash_attn_v3_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    B, H, HQ, N_CTX, D_HEAD,
    stride_q_b, stride_q_h, stride_q_m, stride_q_d,
    stride_k_b, stride_k_h, stride_k_n, stride_k_d,
    stride_v_b, stride_v_h, stride_v_n, stride_v_d,
    stride_o_b, stride_o_h, stride_o_m, stride_o_d,
    scale,
    G: tl.constexpr,
    CAUSAL: tl.constexpr,
    D_HEAD_CONST: tl.constexpr,
    KV_BLOCK: tl.constexpr,
    BQ: tl.constexpr, BV: tl.constexpr,
):
    """
    FlashAttention v3: Q[1, B, HQ, N, D] × K/V[1, B, H, N, D] → O[1, B, HQ, N, D]

    Grid: (cdiv(N, BQ), B * HQ, cdiv(D, BV))
      axis=0: Q sequence tiles
      axis=1: (batch, query_head) pairs
      axis=2: V dimension tiles

    每个 program 处理一个 (Q_tile, head) 对，扫描全部 K/V。
    """
    pid_q = tl.program_id(0)    # Q 序列 tile
    pid_bh = tl.program_id(1)   # (batch, head_Q)
    pid_v = tl.program_id(2)    # V 维度 tile

    batch_idx = pid_bh // HQ
    hq_idx = pid_bh % HQ
    h_idx = hq_idx // G          # GQA: 映射到 KV head

    # ── Q tile: [BQ, D_HEAD] ──────────────────────────────────
    q_offset = batch_idx * stride_q_b + hq_idx * stride_q_h
    p_q = tl.make_block_ptr(
        base=q_ptr + q_offset,
        shape=(N_CTX, D_HEAD),
        strides=(stride_q_m, stride_q_d),
        offsets=(pid_q * BQ, 0),
        block_shape=(BQ, D_HEAD_CONST),
        order=(1, 0),
    )
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)

    # ── O tile: [BQ, BV] ──────────────────────────────────────
    p_o = tl.make_block_ptr(
        base=o_ptr + q_offset,
        shape=(N_CTX, D_HEAD),
        strides=(stride_o_m, stride_o_d),
        offsets=(pid_q * BQ, pid_v * BV),
        block_shape=(BQ, BV),
        order=(1, 0),
    )

    # ── Online softmax state ───────────────────────────────────
    m_i = tl.full([BQ, 1], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BQ, 1], dtype=tl.float32)
    acc = tl.zeros([BQ, BV], dtype=tl.float32)

    # K block uses D_HEAD_CONST for the D dimension (no tiling along D)
    num_kv_blocks = tl.cdiv(N_CTX, KV_BLOCK)

    for kv_block in range(num_kv_blocks):
        kv_offset = batch_idx * stride_k_b + h_idx * stride_k_h

        # K tile: [KV_BLOCK, D_HEAD_CONST]
        p_k = tl.make_block_ptr(
            base=k_ptr + kv_offset,
            shape=(N_CTX, D_HEAD),
            strides=(stride_k_n, stride_k_d),
            offsets=(kv_block * KV_BLOCK, 0),
            block_shape=(KV_BLOCK, D_HEAD_CONST),
            order=(1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))

        # V tile: [KV_BLOCK, BV]
        p_v = tl.make_block_ptr(
            base=v_ptr + kv_offset,
            shape=(N_CTX, D_HEAD),
            strides=(stride_v_n, stride_v_d),
            offsets=(kv_block * KV_BLOCK, pid_v * BV),
            block_shape=(KV_BLOCK, BV),
            order=(1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))

        # QK^T: [BQ, D] @ [KV_BLOCK, D]^T → [BQ, KV_BLOCK]
        b_s = tl.dot(b_q.to(tl.float16), b_k.to(tl.float16).T).to(tl.float32)

        # Causal mask
        if CAUSAL:
            q_pos = pid_q * BQ + tl.arange(0, BQ)[:, None]
            k_pos = kv_block * KV_BLOCK + tl.arange(0, KV_BLOCK)[None, :]
            causal_mask = k_pos <= q_pos
            b_s = tl.where(causal_mask, b_s, float("-inf"))

        # Boundary mask
        k_bound = kv_block * KV_BLOCK + tl.arange(0, KV_BLOCK) < N_CTX
        b_s = tl.where(k_bound[None, :], b_s, float("-inf"))

        # Online softmax
        m_new = tl.maximum(m_i, tl.max(b_s, axis=1, keep_dims=True))
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(tl.exp(b_s - m_new), axis=1, keep_dims=True)
        acc = acc * alpha

        p = tl.exp(b_s - m_new)
        acc += tl.dot(p.to(tl.float16), b_v.to(tl.float16)).to(tl.float32)
        m_i = m_new

    # ── Normalize & store ──────────────────────────────────────
    acc = acc / l_i
    tl.store(p_o, acc.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))


def flash_attention_v3(
    q: torch.Tensor,           # (B, HQ, N, D_HEAD)
    k: torch.Tensor,           # (B, H,  N, D_HEAD)
    v: torch.Tensor,           # (B, H,  N, D_HEAD)
    causal: bool = True,
    scale: float = None,
) -> torch.Tensor:
    B, HQ, N, D_HEAD = q.shape
    _, H, _, _ = k.shape

    if scale is None:
        scale = D_HEAD ** -0.5
    G = HQ // H

    o = torch.empty_like(q)

    grid = lambda meta: (
        triton.cdiv(N, meta["BQ"]),
        B * HQ,
        triton.cdiv(D_HEAD, meta["BV"]),
    )

    flash_attn_v3_kernel[grid](
        q, k, v, o,
        B, H, HQ, N, D_HEAD,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        scale,
        G=G, CAUSAL=causal, D_HEAD_CONST=D_HEAD, KV_BLOCK=32,
    )
    return o


def ref_attention(q, k, v, causal=True, scale=None):
    """PyTorch reference with GQA support."""
    if scale is None:
        scale = q.shape[-1] ** -0.5
    B, HQ, N, D = q.shape
    _, H, _, _ = k.shape
    G = HQ // H

    out = torch.zeros_like(q, dtype=torch.float32)
    for b in range(B):
        for hq in range(HQ):
            h = hq // G
            scores = (q[b, hq].float() @ k[b, h].float().T) * scale
            if causal:
                causal_mask = torch.tril(torch.ones(N, N, device=scores.device))
                scores = scores.masked_fill(causal_mask == 0, float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            out[b, hq] = attn @ v[b, h].float()
    return out.to(q.dtype)


def main():
    print("=" * 60)
    print("03_flash_attention_v3 — Production FlashAttention (Block Ptr)")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    configs = [
        (1, 8, 2, 128, 64, True, "GQA 8Q/2KV D=64 N=128"),
        (1, 8, 2, 256, 64, True, "GQA 8Q/2KV D=64 N=256"),
        (1, 16, 4, 128, 128, True, "GQA 16Q/4KV D=128 N=128"),
        (1, 4, 1, 256, 64, True, "MQA 4Q/1KV D=64 N=256"),
        (2, 8, 2, 128, 64, False, "non-causal GQA 8Q/2KV"),
    ]

    for B, HQ, H, N, D, causal, desc in configs:
        q = torch.randn(B, HQ, N, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

        o_triton = flash_attention_v3(q, k, v, causal=causal)
        o_ref = ref_attention(q, k, v, causal=causal)
        max_diff = (o_triton.float() - o_ref.float()).abs().max().item()

        ms_triton = do_bench(lambda: flash_attention_v3(q, k, v, causal=causal))
        # SDPA doesn't support GQA head broadcasting; expand K/V heads for comparison
        k_mha = k.repeat_interleave(HQ // H, dim=1)
        v_mha = v.repeat_interleave(HQ // H, dim=1)
        ms_sdpa = do_bench(
            lambda: torch.nn.functional.scaled_dot_product_attention(
                q, k_mha, v_mha, is_causal=causal))

        ok = max_diff < 0.05
        print(f"  {desc}: Triton={ms_triton:.4f}ms SDPA={ms_sdpa:.4f}ms "
              f"({ms_sdpa/ms_triton:.2f}x) diff={max_diff:.2e} "
              f"{'OK' if ok else 'FAIL'}")


# PERFORMANCE NOTES
# =================
# Block pointer 在生产 FlashAttention 中的价值:
# 1. Q/K/V/O 全部用 block_ptr → 编译器可推理完整的访问模式
# 2. boundary_check 自动处理序列尾和 D 维的边界
# 3. 3D grid: (Q_tiles, B*HQ, V_tiles) → 充分利用 SM
# 4. GQA: HEAD_RATIO 映射 Q head → KV head
# 5. tl.dot 的双向: QK^T 和 PV 都用 Tensor Core
# 6. Online softmax: rescale 避免存储完整的 N×N attention matrix


if __name__ == "__main__":
    main()
