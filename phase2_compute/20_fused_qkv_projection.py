"""
18_fused_qkv_projection.py — Fused QKV Projection（Transformer Attention 前序）

学习目标:
  - 理解如何将 3 个 MatMul 融合为 1 个
  - 掌握 weight interleaving 的数据布局技巧
  - 学习列分区的 output tiling

标准 Transformer attention:
  Q = x @ W_q     (M×K) @ (K×d) → (M×d)
  K = x @ W_k     (M×K) @ (K×d) → (M×d)
  V = x @ W_v     (M×K) @ (K×d) → (M×d)

Fused 策略:
  将 W_q, W_k, W_v 拼成 (K, 3*d)，一次 matmul 算出 [Q|K|V]
  输出: (M, 3*d)，然后 split 为 Q, K, V

  grid 按 3 个 head group 分:
    axis=1 的 3 份对应 Q/K/V，但可以统一成一个 wider matmul
    然后用 output slice 写入各自的 buffer

  也可以: 3 个 matmul 各自用独立的 grid partition:
    axis=1 分 3 段: [0..d, d..2d, 2d..3d]
    每个段用对应的 weight slice

  本实现用第 2 种: 1 个 kernel，grid axis=1 = cdiv(3*d, BN)
  自然对应 Q/K/V 的输出 tile。

运行: python phase2_compute/18_fused_qkv_projection.py
"""

import torch
import triton
import triton.language as tl
from triton.testing import do_bench


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def fused_qkv_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N, K,
    d_head,              # 单个 Q/K/V 的 head dim
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    out = x @ W  where W = [W_q | W_k | W_v], shape (K, 3*d_head)

    N = 3 * d_head
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc += tl.dot(x, w)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


def fused_qkv_projection(x: torch.Tensor, w_qkv: torch.Tensor,
                         d_head: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    x: (M, K)
    w_qkv: (K, 3 * d_head)
    返回: Q, K, V 各 (M, d_head)
    """
    M, K = x.shape
    K2, N = w_qkv.shape
    assert K == K2 and N == 3 * d_head

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    fused_qkv_kernel[grid](
        x, w_qkv, out,
        M, N, K, d_head,
        x.stride(0), x.stride(1),
        w_qkv.stride(0), w_qkv.stride(1),
        out.stride(0), out.stride(1),
    )
    q, k, v = out.split(d_head, dim=1)
    return q, k, v


def unfused_qkv_projection(x: torch.Tensor, w_q: torch.Tensor,
                           w_k: torch.Tensor, w_v: torch.Tensor):
    """分离的 Q, K, V"""
    q = x @ w_q
    k = x @ w_k
    v = x @ w_v
    return q, k, v


def main():
    print("=" * 60)
    print("18_fused_qkv_projection — Fused QKV Projection")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    configs = [
        (128, 256, 64),       # (M, K, d_head) — 小
        (512, 512, 64),       # 中
        (1024, 768, 64),      # 常见 BERT
        (2048, 1024, 128),    # 大
    ]

    for M, K, d_head in configs:
        x = torch.randn(M, K, device="cuda", dtype=torch.float16)
        w_qkv = torch.randn(K, 3 * d_head, device="cuda", dtype=torch.float16)
        w_q, w_k, w_v = w_qkv.split(d_head, dim=1)

        # Fused
        q_f, k_f, v_f = fused_qkv_projection(x, w_qkv, d_head)

        # Unfused reference
        q_u = x @ w_q
        k_u = x @ w_k
        v_u = x @ w_v

        diff_q = (q_f.float() - q_u.float()).abs().max().item()
        diff_k = (k_f.float() - k_u.float()).abs().max().item()
        diff_v = (v_f.float() - v_u.float()).abs().max().item()
        max_diff = max(diff_q, diff_k, diff_v)

        ms_fused = do_bench(lambda: fused_qkv_projection(x, w_qkv, d_head))
        ms_unfused = do_bench(lambda: unfused_qkv_projection(x, w_q, w_k, w_v))
        ms_cublas = do_bench(lambda: x @ w_qkv)  # cuBLAS wider matmul

        total_flops = 2 * M * K * (3 * d_head)
        tflops_f = total_flops / (ms_fused * 1e-3) / 1e12
        speedup = ms_unfused / ms_fused

        status = "✅" if max_diff < 0.01 else "❌"
        print(f"  M={M}, K={K}, d={d_head}:")
        print(f"    Triton: {ms_fused:.4f}ms  {tflops_f:.1f} TFLOPS")
        print(f"    cuBLAS: {ms_cublas:.4f}ms  "
              f"(fused vs cuBLAS: {ms_cublas/ms_fused:.2f}x, "
              f"unfused vs cuBLAS: {ms_cublas/ms_unfused:.2f}x)  "
              f"diff={max_diff:.2e}  {status}")

    print(f"\n  💡 Fused QKV 的核心优势:")
    print(f"     - 读 X 仅 1 次 (而非 3 次)，节省 2/3 的输入带宽")
    print(f"     - 1 个 kernel launch (而非 3 个)，减少 driver overhead")
    print(f"     - 对 memory-bound 场景 (M 小) 提升更明显")


# PERFORMANCE NOTES
# =================
# - Fused QKV 本质是一个更宽的 GEMM: (M, K) @ (K, 3d) → (M, 3d)
# - 优势: 输入 x 被读 1 次（vs 3 次），节省 HBM 带宽
# - 劣势: 更大的 output tile 意味着更多寄存器使用
# - 对于大 M（长序列），计算量主导 → 带宽节省帮助不大
# - 对于小 M（短序列），memory-bound → 带宽节省是关键
# - [COMPILER] 编译器将 3d 宽的 matmul 视为一个 wider GEMM，
#   不会对 Q/K/V 分别做特殊优化
# - 生产实现（如 FlashAttention 的 fused QKV）还会加入 bias 和 RoPE


if __name__ == "__main__":
    main()
