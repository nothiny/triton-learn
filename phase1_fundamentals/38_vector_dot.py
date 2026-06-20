"""
38_vector_dot.py — Vector Dot Product (Inner Product)

学习目标:
  - 掌握内积的 GPU 实现: elementwise mul + reduction sum
  - 理解 dot product 是 GEMM 的一维特例 (M=1, K=N, N=1):
    dot(a, b) = sum(a[i] * b[i]) = matmul(1×N, N×1)
  - 为理解 matmul 的 S=sum(A[i,k]*B[k,j]) 打基础

数学定义:
  dot(a, b) = sum(a[i] * b[i])  for i in [0, N)

和之前算子的关系:
  - 12_vector_sum: sum(x)
  - 32_mse_loss: sum((p-t)²)
  - 38_vector_dot: sum(a*b)  ← 关键差异: 两个输入相乘再归约

为什么重要:
  - dot product → GEMM 的基础操作: C[i,j] = dot(A[i,:], B[:,j])
  - Self-Attention: Q[i]·K[j] 就是 dot product (scaled by 1/sqrt(d))
  - Cosine similarity: dot(a,b) / (|a|*|b|)

运行: python phase1_fundamentals/38_vector_dot.py
"""

import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch, triton, triton.language as tl
from utils.profiler import bench_compare, print_compare_report


@triton.jit
def dot_kernel(a_ptr, b_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """output[0] = sum(a[i] * b[i])"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # [GPU] 一次 load a 和 b, 在寄存器中相乘
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)

    # [GPU] mul + warp shuffle reduce + shared memory → 标量
    partial = tl.sum(a * b, axis=0)
    tl.atomic_add(output_ptr, partial)


def vector_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Dot product: sum(a[i] * b[i]) — 单次遍历."""
    assert a.shape == b.shape
    n = a.numel()
    dot_sum = torch.zeros(1, device=a.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    dot_kernel[grid](a, b, dot_sum, n, BLOCK_SIZE=1024)
    return dot_sum


def main():
    print("=" * 60)
    print("38_vector_dot — Vector Dot Product")
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.manual_seed(42)

    for name, size in [("small ", 256), ("medium", 65536), ("large ", 1048576),
                        ("xl    ", 16777216)]:
        a = torch.randn(size, device="cuda")
        b = torch.randn(size, device="cuda")
        dot_t = vector_dot(a, b).item()
        dot_r = torch.dot(a, b).item()
        err = abs(dot_t - dot_r) / max(abs(dot_r), 1e-30)
        print(f"  [{name}] size={size:9d}  dot={dot_t:12.4f}/{dot_r:12.4f}  "
              f"rel_err={err:.2e}  {'✅' if err < 1e-5 else '❌'}")

    # Demo: dot product as attention score
    print("\n--- Example: Attention score = Q·K (scaled dot product) ---")
    Q = torch.randn(64, device="cuda")  # query vector, d=64
    K = torch.randn(64, device="cuda")  # key vector, d=64
    d = Q.numel()
    score_t = vector_dot(Q, K).item() / (d ** 0.5)
    score_r = torch.dot(Q, K).item() / (d ** 0.5)
    print(f"  scaled_dot_product = Q·K/sqrt(d) = {score_t:.4f} / {score_r:.4f}")

    print("\n--- Performance ---")
    a = torch.randn(16777216, device="cuda", dtype=torch.float32)
    b = torch.randn(16777216, device="cuda", dtype=torch.float32)
    n = a.numel()
    result = bench_compare(
        {
            "Triton dot product": lambda: vector_dot(a, b),
            "PyTorch torch.dot": lambda: torch.dot(a, b),
        },
        flops=n * 2,          # mul + add
        bytes_accessed=n * 8,  # read a + b
        dtype="fp32",
    )
    print_compare_report(result)


# PERFORMANCE NOTES
# =================
# - Dot product = elementwise mul + reduction sum.
#   Arithmetic intensity = 2 FLOP / 8 bytes = 0.25 FLOP/byte → strongly memory-bound.
# - 和 GEMM 的关系: dot product 是 1D GEMM (M=1, N=1, K=N).
#   GEMM 的 arithmetic intensity 随 K 增大 (M=N=K 时是 O(K) FLOP/byte),
#   所以 GEMM 可以是 compute-bound, 但 dot product 永远是 memory-bound.
# - 优化: 用更大 BLOCK_SIZE (4096) 减少 atomic_add 的次数.

if __name__ == "__main__":
    main()
