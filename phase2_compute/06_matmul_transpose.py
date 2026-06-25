"""
06_matmul_transpose.py — 4种转置组合的 GEMM

学习目标:
  - 理解 4 种转置组合 (NN/NT/TN/TT) 的内存访问模式
  - 掌握 stride 参数如何表达转置
  - 理解 coalesced access 对性能的影响

四种转置组合:
  NN: C = A @ B      (A: M×K, B: K×N)      — 标准 forward GEMM
  NT: C = A @ B^T    (A: M×K, B: N×K)      — 最常用: Q @ K^T (attention)
  TN: C = A^T @ B    (A: K×M, B: K×N)      — 梯度: dX = W^T @ dY
  TT: C = A^T @ B^T  (A: K×M, B: N×K)      — 较罕见

Stride 语义:
  通过调整 stride 参数来"表达"转置，而不需要物理拷贝数据:

  标准 NN: A.stride(0)=K, A.stride(1)=1  → 行优先
          B.stride(0)=N, B.stride(1)=1  → 行优先

  如果 A 是转置 (K×M) 而非 (M×K):
    物理存储: A^T 按 (M, K) 存储
    但逻辑上 A 是 (K, M) — 此时 A.stride(0)=1, A.stride(1)=K
    即: 沿 M 维 stride=1 (列连续), 沿 K 维 stride=M

  在 kernel 中，调整 stride_am 和 stride_ak 即可处理所有情况。

Coalesced Access 分析:
  - NN: A 按 K 维加载 → stride_ak=1 → coalesced ✓
  - NT: A 按 K 维加载 → stride_ak=1 → coalesced ✓
        但 B 按 N 维加载 → stride_bn 可能 ≠ 1 → 可能非 coalesced
  - TN: A 按 M 维加载 → stride_am=1 → coalesced ✓
  - TT: 两种都转置 → 都可以 coalesced 加载（取决于 stride）

运行: python phase2_compute/06_matmul_transpose.py
"""

import torch
import triton
import triton.language as tl
from triton.testing import do_bench


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_transpose_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,  # A: how to step along M and K dimensions
    stride_bk, stride_bn,  # B: how to step along K and N dimensions
    stride_cm, stride_cn,  # C: output strides
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TRANSPOSE_A: tl.constexpr,  # whether A is transposed (logical K×M stored as M×K)
    TRANSPOSE_B: tl.constexpr,  # whether B is transposed (logical N×K stored as K×N)
):
    """
    GEMM kernel supporting all 4 transpose patterns.

    Input:
      A: (M, K) if TRANSPOSE_A=False, else (K, M) if TRANSPOSE_A=True
      B: (K, N) if TRANSPOSE_B=False, else (N, K) if TRANSPOSE_B=True
      C: always (M, N)

    [COMPILER] TRANSPOSE_A/B 是 tl.constexpr，编译器为每种组合生成专用 PTX，
    零运行时开销。
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        # Load A tile: [BLOCK_M, BLOCK_K]
        # 无论是否转置，都需要 (M_tile, K_tile) 的 2D slice
        # stride_am/ak 已经在 Python wrapper 中根据转置标志正确设置
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B tile: [BLOCK_K, BLOCK_N]
        # 同样，stride_bk/bn 在 wrapper 中已正确设置
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


# ==============================================================================
# Python Wrappers — 4 variants
# ==============================================================================


def matmul_nn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """C = A @ B  (A: M×K, B: K×N) — Standard forward GEMM."""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    matmul_transpose_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),  # A: row-major (M, K)
        b.stride(0), b.stride(1),  # B: row-major (K, N)
        c.stride(0), c.stride(1),
        TRANSPOSE_A=False, TRANSPOSE_B=False,
    )
    return c


def matmul_nt(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """C = A @ B^T  (A: M×K, B: N×K → B^T: K×N).

    这是 attention 中最常见的模式:  Q @ K^T
    B 物理存储为 (N, K)，但逻辑上我们需要 B^T = (K, N)。

    关键: B.stride(0) 是沿 N 维的步长, B.stride(1) 是沿 K 维的步长。
    要"转置" B，我们交换 stride 的含义:
      - 沿 K 维加载: stride_bk = B.stride(1) (=1, 连续)
      - 沿 N 维加载: stride_bn = B.stride(0) (=K, 跨行)
    """
    M, K = a.shape
    N, K2 = b.shape  # B is (N, K), stored as row-major
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    matmul_transpose_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),  # A: (M, K) → stride_am=K, stride_ak=1
        b.stride(1), b.stride(0),  # B: (N, K) but we want B^T: stride_bk=1, stride_bn=K
        c.stride(0), c.stride(1),
        TRANSPOSE_A=False, TRANSPOSE_B=True,
    )
    return c


def matmul_tn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """C = A^T @ B  (A: K×M → A^T: M×K, B: K×N).

    Gradient pattern: dX = W^T @ dY
    A 物理存储为 (K, M)，逻辑上需要 A^T = (M, K)。

    类似地，交换 A 的 stride:
      - 沿 M 维加载: stride_am = A.stride(1) (=1, 连续)
      - 沿 K 维加载: stride_ak = A.stride(0) (=M, 跨行)
    """
    K, M = a.shape  # A is (K, M)
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    matmul_transpose_kernel[grid](
        a, b, c, M, N, K,
        a.stride(1), a.stride(0),  # A: (K, M) → want A^T: stride_am=1, stride_ak=M
        b.stride(0), b.stride(1),  # B: (K, N) → stride_bk=N, stride_bn=1
        c.stride(0), c.stride(1),
        TRANSPOSE_A=True, TRANSPOSE_B=False,
    )
    return c


def matmul_tt(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """C = A^T @ B^T  (A: K×M, B: N×K). Both transposed."""
    K, M = a.shape
    N, K2 = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    matmul_transpose_kernel[grid](
        a, b, c, M, N, K,
        a.stride(1), a.stride(0),  # A^T: stride_am=1, stride_ak=M
        b.stride(1), b.stride(0),  # B^T: stride_bk=1, stride_bn=K
        c.stride(0), c.stride(1),
        TRANSPOSE_A=True, TRANSPOSE_B=True,
    )
    return c


# ==============================================================================
# Main
# ==============================================================================


def main():
    print("=" * 70)
    print("06_matmul_transpose — 4 Transpose Variants")
    print("=" * 70)

    torch.manual_seed(42)

    # Test all 4 variants
    variants = [
        ("NN", matmul_nn, "C = A @ B  (standard forward)"),
        ("NT", matmul_nt, "C = A @ B^T (Q @ K^T in attention)"),
        ("TN", matmul_tn, "C = A^T @ B (dX = W^T @ dY)"),
        ("TT", matmul_tt, "C = A^T @ B^T (both transposed)"),
    ]

    for name, fn, desc in variants:
        print(f"\n── {name}: {desc} ──")

        for M, N, K in [(256, 256, 512), (512, 512, 1024), (1024, 1024, 2048)]:
            # Create tensors with shapes matching the variant
            if name == "NN":
                a = torch.randn(M, K, device="cuda", dtype=torch.float16)
                b = torch.randn(K, N, device="cuda", dtype=torch.float16)
                ref = torch.mm(a.float(), b.float()).half()
            elif name == "NT":
                a = torch.randn(M, K, device="cuda", dtype=torch.float16)
                b = torch.randn(N, K, device="cuda", dtype=torch.float16)  # (N,K) for B^T
                ref = torch.mm(a.float(), b.float().T).half()
            elif name == "TN":
                a = torch.randn(K, M, device="cuda", dtype=torch.float16)  # (K,M) for A^T
                b = torch.randn(K, N, device="cuda", dtype=torch.float16)
                ref = torch.mm(a.float().T, b.float()).half()
            else:  # TT
                a = torch.randn(K, M, device="cuda", dtype=torch.float16)  # (K,M)
                b = torch.randn(N, K, device="cuda", dtype=torch.float16)  # (N,K)
                ref = torch.mm(a.float().T, b.float().T).half()

            c = fn(a, b)
            max_diff = (c.float() - ref.float()).abs().max().item()
            tol = max(0.05, 0.005 * (K ** 0.5))
            status = "✅" if max_diff < tol else "❌"
            if M <= 256:  # Only print for smaller sizes
                print(f"  {M}×{N}×{K}: diff={max_diff:.2e} (tol={tol:.1e}) {status}")

    # Performance comparison: all 4 variants at same logical size
    print(f"\n{'='*70}")
    print("Performance: 4 variants at 1024×1024×2048")
    print(f"{'='*70}")

    M, N, K = 1024, 1024, 2048
    a_nn = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b_nn = torch.randn(K, N, device="cuda", dtype=torch.float16)
    a_t = torch.randn(K, M, device="cuda", dtype=torch.float16)
    b_t = torch.randn(N, K, device="cuda", dtype=torch.float16)

    for name, fn, a_in, b_in in [
        ("NN", matmul_nn, a_nn, b_nn),
        ("NT", matmul_nt, a_nn, b_t),
        ("TN", matmul_tn, a_t, b_nn),
        ("TT", matmul_tt, a_t, b_t),
    ]:
        ms = do_bench(lambda fn=fn, a_in=a_in, b_in=b_in: fn(a_in, b_in))
        tflops = (2 * M * N * K) / (ms * 1e-3) / 1e12
        print(f"  {name}: {ms:.3f}ms  {tflops:.1f} TFLOPS")

    # Coalesced access analysis
    print(f"\n{'='*70}")
    print("Coalesced Access Analysis")
    print(f"{'='*70}")
    print("""
  合并访问 (coalesced access) 条件:
    同一 warp 内的连续线程访问连续的内存地址。
    stride=1 的维度保证 coalesced access。

  NN:  A stride_ak=1 (K连续) → coalesced ✓
       B stride_bn=1 (N连续) → coalesced ✓
       → 最优 memory access

  NT:  A stride_ak=1 (K连续) → coalesced ✓
       B stride_bn=K (跨行, N不连续) → 部分 coalesced
       → 常见于 attention (Q @ K^T)

  TN:  A stride_am=1 (M连续) → coalesced ✓ (但加载模式是沿M而非沿K)
       B stride_bn=1 (N连续) → coalesced ✓
       → 常见于 gradient (W^T @ dY)

  TT:  A stride_am=1 → coalesced ✓
       B stride_bn=K → 部分 coalesced
       → 较少使用
  """)


# PERFORMANCE NOTES
# =================
# - 核心洞察: 转置只是 stride 交换，不涉及数据拷贝
# - 通过修改 stride_am/stride_ak 和 stride_bk/stride_bn 即可表达所有 4 种组合
# - [COMPILER] TRANSPOSE_A/B 是 tl.constexpr:
#   - 编译器为每种组合生成专用的 PTX
#   - 但实际生成的代码是相同的（因为 stride 不同在 Python wrapper 中处理）
#   - TRANSPOSE 标志目前主要用于文档和未来可能的优化
# - NT (Q @ K^T) 是最重要的模式:
#   - 每个 transformer forward pass 都有至少一个 Q @ K^T
#   - K^T 的 stride 模式会导致对 B 的非 coalesced 访问
#   - 但在 Flash Attention 中，Q 和 K 都分块加载到 shared memory
#   - shared memory 对小 tile (32×128) 的访问不存在 coalescing 问题
# - 性能差异主要来自:
#   - 非 coalesced global memory 访问 → extra L2 transactions
#   - 对于大 tile，shared memory staging 隐藏了大部分差异
# - PyTorch 中显式 .T 操作会创建 view（不拷贝），但 stride 改变


if __name__ == "__main__":
    main()
