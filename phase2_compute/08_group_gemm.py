"""
15_group_gemm.py — Grouped GEMM (MoE 风格的独立分组矩阵乘)

学习目标:
  - 掌握 variable-size group matmul 的调度策略
  - 理解 MoE (Mixture of Experts) 的计算模式
  - 学习如何通过 pointer arithmetic 在 kernel 内定位不同 group 的数据

场景（MoE forward 简化版）:
  输入 x: (M, K)，有 num_groups 个 expert，每个 expert 处理一部分 embedding dim。
  Expert i 的权重: W_i: (K, N_i)，其中 N = sum(N_i)。
  输出: (M, N) = concat_i[ x @ W_i ]

  实际 MoE 还有 token routing（每个 token 被路由到 top-2 expert），这里先简化：
  每个 expert 处理相同输入 x，输出各自负责的 columns。

核心技巧:
  用 offsets 数组记录每个 expert weight 的起始列索引，
  kernel 内根据 group_id 定位到对应的 weight slice。

运行: python phase2_compute/15_group_gemm.py
"""

import torch
import triton
import triton.language as tl
from triton.testing import do_bench


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
    ],
    key=["M", "K", "num_groups"],
)
@triton.jit
def group_gemm_kernel(
    x_ptr,              # (M, K) 共享输入
    w_ptr,              # (K, N) 拼接的权重 [W_0 | W_1 | ... | W_{G-1}]
    out_ptr,            # (M, N) 拼接的输出
    w_col_offsets_ptr,  # [G+1]: w_col_offsets[g] = group g 的起始列索引
    M, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    每个 program 计算某个 (pid_m, group_id) tile。

    Grid: (cdiv(M, BLOCK_M), num_groups)
    - axis=1 表示 group/ expert 编号
    - axis=0 表示 M 维度的 tile
    """
    pid_m = tl.program_id(axis=0)
    group_id = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # 读取该 group 的列偏移
    col_start = tl.load(w_col_offsets_ptr + group_id)
    col_end = tl.load(w_col_offsets_ptr + group_id + 1)
    group_n = col_end - col_start  # 该 expert 的输出宽度 N_i

    # 该 group 的 N 维 tile 数
    num_pid_n = tl.cdiv(group_n, BLOCK_N)
    # 该 group 的 W 偏移
    w_group_ptr = w_ptr + col_start * stride_wn
    out_group_ptr = out_ptr + col_start * stride_on

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # 对 K 维和 N 维迭代
    for pid_n in range(num_pid_n):
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < group_n

        # 重置累加器（每个 N tile 独立）
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        for k in range(0, K, BLOCK_K):
            offs_k = k + tl.arange(0, BLOCK_K)

            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
            x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
            x = tl.load(x_ptrs, mask=x_mask, other=0.0)

            w_ptrs = w_group_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
            w_mask = (offs_k[:, None] < K) & n_mask[None, :]
            w = tl.load(w_ptrs, mask=w_mask, other=0.0)

            acc += tl.dot(x, w)

        out_ptrs = out_group_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        out_mask = (offs_m[:, None] < M) & n_mask[None, :]
        tl.store(out_ptrs, acc, mask=out_mask)


def group_gemm(x: torch.Tensor, weights: list[torch.Tensor]) -> torch.Tensor:
    """
    x: (M, K) — 共享输入
    weights: list of (K, N_i) tensors — 每个 group/expert 的权重

    返回: (M, N) where N = sum(N_i)
    """
    M, K = x.shape
    num_groups = len(weights)

    # 拼接权重: (K, N)
    w_cat = torch.cat(weights, dim=1)
    N = w_cat.shape[1]

    # 列偏移: [0, N_0, N_0+N_1, ...]
    offsets = [0]
    for w in weights:
        offsets.append(offsets[-1] + w.shape[1])
    w_col_offsets = torch.tensor(offsets, device=x.device, dtype=torch.int32)

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        num_groups,
    )

    group_gemm_kernel[grid](
        x, w_cat, out, w_col_offsets,
        M, K,
        x.stride(0), x.stride(1),
        w_cat.stride(0), w_cat.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


def main():
    print("=" * 60)
    print("15_group_gemm — Grouped GEMM (MoE-style)")
    print("=" * 60)

    for M, K, num_groups in [(256, 256, 4), (512, 256, 8), (1024, 512, 4)]:
        x = torch.randn(M, K, device="cuda", dtype=torch.float16)

        # 每个 expert 处理不同的 N_i
        import random
        random.seed(42)
        ns = [random.randint(32, 128) for _ in range(num_groups)]
        weights = [torch.randn(K, n, device="cuda", dtype=torch.float16) for n in ns]

        # Triton group GEMM
        out_triton = group_gemm(x, weights)

        # PyTorch reference: concat + single matmul
        w_cat = torch.cat(weights, dim=1)
        out_ref = x @ w_cat

        max_diff = (out_triton.float() - out_ref.float()).abs().max().item()
        ms = do_bench(lambda: group_gemm(x, weights))

        total_flops = 2 * M * K * sum(ns)
        tflops = total_flops / (ms * 1e-3) / 1e12

        status = "✅" if max_diff < 0.01 else "❌"
        print(f"  M={M}, K={K}, groups={num_groups}, Ns={ns}: "
              f"{ms:.4f}ms  {tflops:.1f} TFLOPS  diff={max_diff:.2e}  {status}")

    # 对比: group GEMM vs 拼接后单次 matmul
    print("\n  💡 Group GEMM 的优势不在于性能（拼接后单次 matmul 更快），")
    print("     而在于避免拼接/拆分开销。在 MoE 中每个 token 走到不同 expert，")
    print("     group GEMM 可以直接在 kernel 内做 routing + gather/scatter。")


# PERFORMANCE NOTES
# =================
# - 本实现的 grid 是 2D: (M tiles, num_groups)，每个 program 处理一个 (M tile, expert) 对
# - 主要开销: 小 expert 时 grid 利用率不均 (N 小的 expert 产生很少的 tile)
# - 优化方向:
#   1. 合并小 experts 到同一个 program → load balancing
#   2. Token routing (每个 token 只计算 top-2 experts) → 减少无效计算
#   3. 对 N 维特别大的 expert 做 split-K
# - 与标准 GEMM 的区别: 不同 group 处理不同 weight → 不共享 L2 cache 中的 weight 数据


if __name__ == "__main__":
    main()
