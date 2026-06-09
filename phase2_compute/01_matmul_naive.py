"""
01_matmul_naive.py — 最简单的分块 GEMM（不用 shared memory）

学习目标：
  - 理解 GEMM 的 tiling 逻辑
  - 掌握 M/N/K 维度的分块策略
  - 学会计算 TFLOPS = 2 * M * N * K / time

运行: python phase2_compute/01_matmul_naive.py
"""

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_naive_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,  # A 的 stride (M, K)
    stride_bk, stride_bn,  # B 的 stride (K, N)
    stride_cm, stride_cn,  # C 的 stride (M, N)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Naive tiled MatMul: C[m, n] = sum_k A[m, k] * B[k, n]

    不使用 shared memory，直接从 HBM 读取 A 和 B。
    [COMPILER] Triton 可能会自动缓存部分数据到 L1/L2。

    每个 program 负责计算 C 的一个 [BLOCK_M × BLOCK_N] tile。
    """
    # program_id: (pid_m, pid_n) → 对应 C 的第几块 tile
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # 当前 program 负责的 C tile 范围
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # 累加器: [BLOCK_M, BLOCK_N]
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # 沿 K 维迭代（K 可能远超 BLOCK_K）
    for k in range(0, K, BLOCK_K):
        # A tile: [BLOCK_M, BLOCK_K]  从 (offs_m, k) 位置读取
        offs_k = k + tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # B tile: [BLOCK_K, BLOCK_N]  从 (k, offs_n) 位置读取
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # 矩阵乘: acc += A_tile @ B_tile
        # tl.dot 映射到 Tensor Core MMA 指令
        acc += tl.dot(a, b)

    # 写回 C
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def matmul_naive(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Naive tiled MatMul wrapper"""
    assert a.dim() == 2 and b.dim() == 2
    assert a.shape[1] == b.shape[0], f"dim mismatch: {a.shape} @ {b.shape}"

    M, K = a.shape
    K2, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    matmul_naive_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=64, BLOCK_N=128, BLOCK_K=32,
    )
    return c


def main():
    print("=" * 60)
    print("01_matmul_naive — Tiled GEMM (no shared memory)")
    print("=" * 60)

    M, N, K = 1024, 1024, 1024
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)

    # 正确性
    c_triton = matmul_naive(a, b)
    c_torch = torch.mm(a, b)
    max_diff = (c_triton.float() - c_torch.float()).abs().max().item()
    print(f"  Max diff: {max_diff:.6e}  {'✅' if max_diff < 0.01 else '❌'}")

    # 性能
    n_iter = 100
    for _ in range(10):
        matmul_naive(a, b)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        matmul_naive(a, b)
    end.record()
    torch.cuda.synchronize()
    triton_ms = start.elapsed_time(end) / n_iter

    tflops = (2 * M * N * K) / (triton_ms * 1e-3) / 1e12
    print(f"  Time: {triton_ms:.4f} ms")
    print(f"  TFLOPS: {tflops:.2f}")


# PERFORMANCE NOTES
# =================
# - 这个版本从 HBM 读 A 和 B，没有 shared memory 缓存
# - 算术强度: (2*M*N*K) FLOP / ((M*K + K*N + M*N) * dtype_size) bytes
#   - 对于大矩阵: 趋于 O(N) FLOP/O(N^2) bytes = O(N) 算术强度
# - 瓶颈: HBM 带宽（没有 shared memory 缓存复用）
# - 下一步: 02_matmul_tiled.py 加入 shared memory，大幅提升性能
# - [COMPILER] tl.dot(a, b) 自动映射为 MMA 指令。Triton 根据 BLOCK 尺寸
#   选择合适的 MMA 布局 (MmaEncodingAttr v1/v2/v3)


if __name__ == "__main__":
    main()
