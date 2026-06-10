"""
13_vector_max.py — Vector Max Reduction kernel

学习目标:
  - 掌握 tl.max reduction 操作
  - 理解 max 和 sum 在 warp shuffle 层面的实现差异
  - 学习 reduction 的 identity element 概念 (max 的 identity 是 -inf)

数学公式:
  result = max(x[i])  for i in [0, N)

运行: python phase1_fundamentals/13_vector_max.py
"""

import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch, triton, triton.language as tl
from utils.profiler import bench_compare, print_compare_report


@triton.jit
def max_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """1D max reduction: output[0] = max(x)"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=float("-inf"))
    partial = tl.max(x, axis=0)
    tl.atomic_max(output_ptr, partial)


def vector_max(x: torch.Tensor) -> torch.Tensor:
    output = torch.full((1,), float("-inf"), device=x.device, dtype=torch.float32)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    max_kernel[grid](x, output, n, BLOCK_SIZE=1024)
    return output


def main():
    print("=" * 60)
    print("13_vector_max — Vector Max Reduction")
    print("=" * 60)
    if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.manual_seed(42)
    for name, size in [("small", 1024), ("medium", 65536), ("large", 1048576)]:
        x = torch.randn(size, device="cuda")
        out_triton = vector_max(x).item()
        out_torch = x.max().item()
        diff = abs(out_triton - out_torch)
        print(f"  [{name}] size={size} triton={out_triton:.4f} torch={out_torch:.4f} diff={diff:.2e} {'✅' if diff<1e-5 else '❌'}")
    print("\n--- Performance ---")
    x = torch.randn(16777216, device="cuda", dtype=torch.float32)
    n = x.numel()
    result = bench_compare({"Triton (ours)": lambda: vector_max(x), "PyTorch (ref)": lambda: x.max()}, flops=n, bytes_accessed=n*4, dtype="fp32")
    print_compare_report(result)

# PERFORMANCE NOTES
# =================
# - Max reduction 和 sum reduction 性能相似 (都是 memory-bound)
# - tl.atomic_max 是 Triton 3.x 新增的原语, 比手动 CAS loop 更高效
# - identity element: max 用 -inf, sum 用 0 — 确保跨 block 聚合时初始值不影响结果

if __name__ == "__main__": main()
