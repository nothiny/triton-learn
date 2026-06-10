"""
12_vector_sum.py — Vector Sum Reduction kernel

学习目标:
  - 掌握 Triton 中最基础的 reduction 操作: tl.sum
  - 理解 block-level reduction → warp shuffle → shared memory 的编译过程
  - 为后续 Softmax、LayerNorm 的 reduction 部分打基础

数学公式:
  result = sum(x[i])  for i in [0, N)

运行: python phase1_fundamentals/12_vector_sum.py
"""

import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch, triton, triton.language as tl
from utils.profiler import bench_compare, print_compare_report


@triton.jit
def sum_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """1D sum reduction: output[0] = sum(x)"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    partial = tl.sum(x, axis=0)
    tl.atomic_add(output_ptr, partial)


def vector_sum(x: torch.Tensor) -> torch.Tensor:
    output = torch.zeros(1, device=x.device, dtype=torch.float32)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    sum_kernel[grid](x, output, n, BLOCK_SIZE=1024)
    return output


def main():
    print("=" * 60)
    print("12_vector_sum — Vector Sum Reduction")
    print("=" * 60)
    if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.manual_seed(42)
    for name, size in [("small", 1024), ("medium", 65536), ("large", 1048576)]:
        x = torch.randn(size, device="cuda")
        out_triton = vector_sum(x).item()
        out_torch = x.sum().item()
        diff = abs(out_triton - out_torch)
        print(f"  [{name}] size={size} triton={out_triton:.4f} torch={out_torch:.4f} diff={diff:.2e} {'✅' if diff<1e-3 else '❌'}")
    print("\n--- Performance ---")
    x = torch.randn(16777216, device="cuda", dtype=torch.float32)
    n = x.numel()
    result = bench_compare({"Triton (ours)": lambda: vector_sum(x), "PyTorch (ref)": lambda: x.sum()}, flops=n, bytes_accessed=n*4, dtype="fp32")
    print_compare_report(result)

# PERFORMANCE NOTES
# =================
# - Sum reduction 是 memory-bound: 1 FLOP / 4 bytes read
# - tl.sum(x, axis=0) → warp shuffle (寄存器) → shared memory → atomic_add (跨 block)
# - atomic_add 是跨 block 聚合的瓶颈, 但对 ~100 个 block 影响不大

if __name__ == "__main__": main()
