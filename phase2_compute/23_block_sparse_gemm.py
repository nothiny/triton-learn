"""
20_block_sparse_gemm.py — Block-Sparse GEMM with Predefined Mask

学习目标:
  - 理解 2:4 结构化稀疏和 block-sparse 的区别
  - 掌握用 block mask 跳过零块的计算调度
  - 学习稀疏计算中 mask 与 tiling 的交互

原理:
  标准 GEMM: C = A @ B，对 (M, K) 的每个 block 都做 MMA
  Block-sparse: 通过预定义的 block_mask 只对非零 block 做 MMA

  本实现:
    - 使用标准 2D grid (M tiles, N tiles)，与 dense GEMM 相同
    - 在 K 维迭代时，通过 block_mask 跳过零 block
    - block_mask: (M//BM, K//BK) 的 bool 矩阵

  加速比 ≈ 1 / density（密度越低越快）

运行: python phase2_compute/20_block_sparse_gemm.py
"""

import torch
import triton
import triton.language as tl
from triton.testing import do_bench


@triton.jit
def block_sparse_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    block_mask_ptr,     # (M_blocks, K_blocks) — bool/int: 1 = non-zero block
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    M_blocks,           # = cdiv(M, BLOCK_M)
    K_blocks,           # = cdiv(K, BLOCK_K)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    C = sparse_A @ dense_B

    使用标准 2D grid (M tiles, N tiles)。
    每个 program 沿 K 维迭代，通过 block_mask 跳过零 block。
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # 沿 K 维迭代，只计算非零 block
    for k_block in range(K_blocks):
        # 检查 mask: block_mask[pid_m, k_block]
        is_nonzero = tl.load(block_mask_ptr + pid_m * K_blocks + k_block)

        if is_nonzero:
            offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)

            a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)

            b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
            b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)

            acc += tl.dot(a, b)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def make_block_sparse(matrix: torch.Tensor, block_m: int, block_k: int,
                      sparsity: float) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    沿 K 维的 block mask（M 维也在 tile 粒度上 mask）。

    返回: (sparse_A, block_mask, M_blocks, K_blocks, density)
    """
    M, K = matrix.shape
    M_blocks = triton.cdiv(M, block_m)
    K_blocks = triton.cdiv(K, block_k)

    # 每个 (M tile, K block) 的 Frobenius norm
    block_norms = torch.zeros(M_blocks, K_blocks, device=matrix.device)
    for i in range(M_blocks):
        m_start = i * block_m
        m_end = min(m_start + block_m, M)
        for j in range(K_blocks):
            k_start = j * block_k
            k_end = min(k_start + block_k, K)
            tile = matrix[m_start:m_end, k_start:k_end]
            block_norms[i, j] = tile.float().norm()

    num_keep = max(1, int(M_blocks * K_blocks * (1 - sparsity)))
    _, topk_idx = torch.topk(block_norms.flatten(), num_keep)
    block_mask = torch.zeros(M_blocks * K_blocks, dtype=torch.int32, device=matrix.device)
    block_mask[topk_idx] = 1
    block_mask = block_mask.reshape(M_blocks, K_blocks)

    # Mask 掉零 block
    sparse_A = matrix.clone()
    for j in range(K_blocks):
        k_start = j * block_k
        k_end = min(k_start + block_k, K)
        for i in range(M_blocks):
            if block_mask[i, j] == 0:
                m_start = i * block_m
                m_end = min(m_start + block_m, M)
                sparse_A[m_start:m_end, k_start:k_end] = 0

    density = block_mask.sum().item() / (M_blocks * K_blocks)
    return sparse_A, block_mask, M_blocks, K_blocks, density


def block_sparse_gemm(a: torch.Tensor, b: torch.Tensor,
                      block_mask: torch.Tensor,
                      M_blocks: int, K_blocks: int,
                      BM: int, BN: int, BK: int) -> torch.Tensor:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))

    block_sparse_gemm_kernel[grid](
        a, b, c, block_mask,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        M_blocks, K_blocks,
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
    )
    return c


def main():
    print("=" * 60)
    print("20_block_sparse_gemm — Block-Sparse GEMM")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 较小的测试以保证 block mask 粒度合理
    # BM 需与 make_block_sparse 保持一致
    configs = [
        (256, 256, 256, 0.5, 32, 64, 128),    # (M,N,K,sparsity,BK,BM,BN)
        (256, 256, 512, 0.75, 32, 64, 128),
        (512, 512, 512, 0.5, 32, 64, 128),
        (512, 512, 1024, 0.875, 64, 64, 128),
    ]

    for M, N, K, target_sparsity, BK, BM, BN in configs:
        a_dense = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)

        a_sparse, block_mask, M_blocks, K_blocks, density = \
            make_block_sparse(a_dense, BM, BK, target_sparsity)

        # Block-sparse GEMM (固定 block 尺寸以匹配 mask)
        c_sparse = block_sparse_gemm(a_sparse, b, block_mask,
                                     M_blocks, K_blocks, BM, BN, BK)

        # Reference
        c_ref = a_sparse @ b

        max_diff = (c_sparse.float() - c_ref.float()).abs().max().item()

        ms_sparse = do_bench(
            lambda: block_sparse_gemm(a_sparse, b, block_mask,
                                      M_blocks, K_blocks, BM, BN, BK))
        ms_dense_triton = do_bench(
            lambda: block_sparse_gemm(a_dense, b,
                                      torch.ones(M_blocks, K_blocks, dtype=torch.int32, device="cuda"),
                                      M_blocks, K_blocks, BM, BN, BK))
        ms_cublas = do_bench(lambda: a_dense @ b)

        status = "✅" if max_diff < 0.01 else "❌"
        print(f"  {M}×{K}×{N} BK={BK} sparsity={target_sparsity:.0%} "
              f"(actual density={density:.1%}):")
        print(f"    sparse={ms_sparse:.4f}ms  dense(Triton)={ms_dense_triton:.4f}ms  "
              f"cuBLAS={ms_cublas:.4f}ms  "
              f"sparse/cuBLAS={ms_cublas/ms_sparse:.2f}x  diff={max_diff:.2e}  {status}")

    print(f"\n  💡 使用 2D grid + intra-kernel mask 的方案避免了 atomic_add 的复杂度")
    print(f"     - 优点: 实现简单，正确性容易保证")
    print(f"     - 缺点: grid 大小不随 sparsity 变化，对极高 sparsity 不太高效")


# PERFORMANCE NOTES
# =================
# - 每个 program 沿 K 维迭代时检查 mask 并跳过零 block
# - Mask load (int32, coalesced) 的 overhead 远小于 MMA 的节省
# - 对低 sparsity (50%): mask 检查开销可忽略，但节省也不大
# - 对高 sparsity (87.5%): 跳过 7/8 的 MMA → ~8x 理论加速
# - 实际加速受限于:
#   1. Mask load 开销（但很小）
#   2. Grid 大小不变 → 稀疏 program 也占用 SM
#   3. Shared memory 仍然分配（因为 block 大小不变）
# - 与 2:4 结构化稀疏的区别:
#   - 2:4: 硬件原生支持 (Ampere+ Sparse Tensor Cores)，每 4 个元素 2 个非零
#   - Block-sparse: 以 block 为粒度，灵活但无硬件加速


if __name__ == "__main__":
    main()
