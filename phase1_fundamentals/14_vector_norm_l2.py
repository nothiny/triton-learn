"""
14_vector_norm_l2.py — L2 Vector Norm kernel

学习目标:
  - 掌握 compute + reduction 的组合模式
  - 理解 sqrt(sum(x²)) 的数值稳定性考虑
  - 学会用 atomic_add 做跨 block 聚合

数学公式:
  L2_norm(x) = sqrt(sum(x[i]²))

应用场景:
  - Gradient clipping: clip_grad_norm_ 的第一步就是算 grad 的 L2 norm
  - Weight decay 正则化
  - Attention 中的 Q·K 归一化

运行: python phase1_fundamentals/14_vector_norm_l2.py
"""

import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch, triton, triton.language as tl
from utils.profiler import bench_compare, print_compare_report


@triton.jit
def l2_norm_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """output[0] = sqrt(sum(x²))"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    partial = tl.sum(x * x, axis=0)
    tl.atomic_add(output_ptr, partial)


def vector_norm_l2(x: torch.Tensor) -> torch.Tensor:
    sq_sum = torch.zeros(1, device=x.device, dtype=torch.float32)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    l2_norm_kernel[grid](x, sq_sum, n, BLOCK_SIZE=1024)
    return torch.sqrt(sq_sum)


def main():
    print("=" * 60)
    print("14_vector_norm_l2 — L2 Vector Norm")
    print("=" * 60)
    if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.manual_seed(42)
    for name, size in [("small", 256), ("medium", 4096), ("large", 65536)]:
        x = torch.randn(size, device="cuda")
        out_triton = vector_norm_l2(x).item()
        out_torch = x.norm(p=2).item()
        diff = abs(out_triton - out_torch)
        print(f"  [{name}] size={size} triton={out_triton:.4f} torch={out_torch:.4f} diff={diff:.2e} {'✅' if diff<1e-3 else '❌'}")
    print("\n--- Performance ---")
    x = torch.randn(16777216, device="cuda", dtype=torch.float32)
    n = x.numel()
    result = bench_compare({"Triton (ours)": lambda: vector_norm_l2(x), "PyTorch (ref)": lambda: x.norm(p=2)}, flops=n*3, bytes_accessed=n*4, dtype="fp32")
    print_compare_report(result)

# PERFORMANCE NOTES
# =================
# - L2 norm = compute(x²) + reduction(sum) + post-processing(sqrt)
# - sqrt 在 CPU/Python 端做 (标量操作, 不占 GPU)
# - 和 15_welford_mean_var 的区别: 这里不需要 in-block 计算, 只需全局 sum(x²)

if __name__ == "__main__": main()
