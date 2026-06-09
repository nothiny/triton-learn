"""
03_matmul_autotuned.py — 扩展 autotuning 搜索空间的 MatMul（含 GROUP_M swizzling）

学习目标：
  - 理解 GROUP_M swizzling 如何改善 L2 cache 利用率
  - 掌握大规模 autotune 搜索空间的构建
  - 学会使用 triton.testing.do_bench 做标准化 benchmark

运行: python phase2_compute/03_matmul_autotuned.py
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": m, "BLOCK_N": n, "BLOCK_K": k, "GROUP_M": g},
                       num_warps=w, num_stages=s)
        for m in [64, 128, 256]
        for n in [64, 128, 256]
        for k in [32, 64]
        for g in [1, 4, 8]
        for w in [4, 8]
        for s in [2, 3, 4]
        # Prune invalid combos
        if not (w == 4 and s > 2)          # num_warps=4 时 stage 不宜太多
        if not (w == 4 and m * n > 128 * 128)  # warp 太少时不处理大 tile
        if not (g > 1 and m < 128)         # GROUP_M 对小 tile 意义不大
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_autotuned_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Autotuned GEMM with GROUP_M swizzling for improved L2 cache locality.

    GROUP_M swizzling 原理:
      默认 grid 调度 (pid_m, pid_n) 按 row-major 排列 program。
      当 M 很大时，相邻的 program 处理相邻的 M 行，但共用相同 N 列。
      这意味着它们在 K 维迭代时会访问 B 的相邻区域 → L2 cache 友好。

      GROUP_M 参数将 program 分组：
        - 每 GROUP_M 个 M 行分为一组
        - 组内按列优先重新编号 program
        - 让相邻 program 处理同一 M 行但不同 N 列
        - 这样 A 的同一区域被多个 program 复用 → 更好的 L2 命中率

    [COMPILER] GROUP_M 只影响 grid 调度（program_id 的映射），
              不改变 kernel 内部计算逻辑。
    """
    # ---- GROUP_M swizzling: 重新映射 program_id ----
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n

    # 当前 program 属于第几组
    group_id = pid // num_pid_in_group
    # 组内第一个 program 的全局 M 偏移
    first_pid_m = group_id * GROUP_M
    # 组内 program 数量（最后一组可能不满）
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    # pid 在组内的偏移
    pid_in_group = pid % num_pid_in_group

    # 组内列优先映射: pid_in_group → (local_m, pid_n)
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    # ---- 正常 tiled GEMM 逻辑 ----
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # 累加器: [BLOCK_M, BLOCK_N]，使用 fp32 避免精度损失
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # 沿 K 维迭代
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        # Load A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        # [COMPILER] num_stages > 1 时，Triton 编译器自动插入 cp.async 做
        # 异步 prefetch，实现 software pipelining
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B tile: [BLOCK_K, BLOCK_N]
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # MMA: acc += a @ b
        # tl.dot 自动映射到 Tensor Core MMA 指令
        # [COMPILER] Triton 根据 BLOCK_M/BLOCK_N/BLOCK_K 自动选择
        # MmaEncodingAttr v1 (Ampere) 或 v2/v3 (Hopper)
        acc += tl.dot(a, b)

    # Store C tile: [BLOCK_M, BLOCK_N]
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def matmul_autotuned(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Autotuned GEMM with GROUP_M swizzling."""
    assert a.dim() == 2 and b.dim() == 2
    assert a.shape[1] == b.shape[0], f"dim mismatch: {a.shape} @ {b.shape}"

    M, K = a.shape
    K2, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
    )

    matmul_autotuned_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


def main():
    print("=" * 60)
    print("03_matmul_autotuned — Autotuned GEMM with GROUP_M swizzling")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    sizes = [
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ]

    for M, N, K in sizes:
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)

        # Correctness
        c_triton = matmul_autotuned(a, b)
        c_torch = torch.mm(a, b)
        max_diff = (c_triton.float() - c_torch.float()).abs().max().item()

        # Performance
        n_iter = 100
        for _ in range(25):
            matmul_autotuned(a, b)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_iter):
            matmul_autotuned(a, b)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / n_iter
        tflops = (2 * M * N * K) / (ms * 1e-3) / 1e12

        status = "✅" if max_diff < 0.01 else "❌"
        print(f"  {M}×{N}×{K}: {ms:.4f}ms  {tflops:.1f} TFLOPS  "
              f"diff={max_diff:.2e}  {status}")

    print(f"\n  💡 The autotuner searched "
          f"{len(matmul_autotuned_kernel.configs)} configs. "
          f"Compare against 02_matmul_tiled.py to see the benefit of "
          f"GROUP_M swizzling and larger search space.")


# PERFORMANCE NOTES
# =================
# - GROUP_M swizzling 主要改善中等规模 GEMM (M=1024-4096) 的 L2 cache 命中率
# - 对很小的 GEMM (M<256): L2 命中率本来就不高，swizzling 收益不大
# - 对很大的 GEMM (M>8192): 每个 program 只处理一小部分，swizzling 帮助有限
# - [COMPILER] 这个实现的 autotune 空间有 3×3×2×3×2×3 = 324 种可能配置，
#   但经过 prune 后约 100-150 种。第一次运行会比较慢（JIT 每个配置都要编译），
#   后续运行命中 cache 就快了。
# - 与 02_matmul_tiled.py 的关键区别:
#   1. GROUP_M swizzling — 更好的 L2 cache 局部性
#   2. 更大的搜索空间 — 更多 BLOCK 尺寸组合
#   3. 1D grid — 通过 GROUP_M 映射到 2D tile 空间


if __name__ == "__main__":
    main()
