"""
40_concat.py — Concatenation of Two 1D Tensors

学习目标:
  - 掌握多输入拼接 (concat) 的 GPU 实现
  - 理解两个不同 base pointer + offset 的访存模式
  - 学习条件分支 (tl.where) 在 kernel 中的应用

数学定义:
  concat(a, b) = [a[0], ..., a[N-1], b[0], ..., b[M-1]]

GPU 实现策略:
  - 每个 program 按全局 absolute position 处理一段连续输出
  - 根据 position 判断读 a 还是 b
  - 两种输入也可能来自同一个 tensor 的不同区域 (如 chunk)

和之前算子的关系:
  - 05_fused_relu_bias: 两个输入 (x + bias), 对应位置相加
  - 40_concat: 两个输入, 不同位置拼接 (不重叠)

运行: python phase1_fundamentals/40_concat.py
"""

import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch, triton, triton.language as tl
from utils.profiler import bench_compare, print_compare_report


@triton.jit
def concat_kernel(a_ptr, b_ptr, output_ptr,
                   n_a, n_b, n_total,
                   BLOCK_SIZE: tl.constexpr):
    """
    output = concat(a, b).
    每个 program 输出一段连续数据, 判断来自 a 还是 b.
    """
    pid = tl.program_id(0)
    out_start = pid * BLOCK_SIZE
    out_offsets = out_start + tl.arange(0, BLOCK_SIZE)
    mask = out_offsets < n_total

    # [GPU] 判断每个元素来自 a 还是 b
    # from_a: 位置 < n_a  → 读 a[position]
    # from_b: 位置 >= n_a → 读 b[position - n_a]
    in_a = out_offsets < n_a
    a_val = tl.load(a_ptr + out_offsets, mask=mask & in_a, other=0.0)
    b_val = tl.load(b_ptr + (out_offsets - n_a), mask=mask & (out_offsets >= n_a),
                    other=0.0)

    # [GPU] 两路选择 — 无分歧 (因为每路都有, 用加法合并)
    # tl.where 在这里不是 "if-else", 而是 CMOV (条件移动),
    # 两个 load 都会执行, 结果选其中一个
    result = a_val + b_val  # 一个为 0, 另一个为实际值

    tl.store(output_ptr + out_offsets, result, mask=mask)


def concat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Concatenate two 1D tensors."""
    n_a, n_b = a.numel(), b.numel()
    n_total = n_a + n_b
    output = torch.empty(n_total, device=a.device, dtype=a.dtype)
    grid = lambda meta: (triton.cdiv(n_total, meta["BLOCK_SIZE"]),)
    concat_kernel[grid](a, b, output, n_a, n_b, n_total, BLOCK_SIZE=1024)
    return output


def main():
    print("=" * 60)
    print("40_concat — 1D Concatenation")
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.manual_seed(42)

    for name, (na, nb) in [("small ", (128, 256)), ("medium", (4096, 8192)),
                             ("uneven", (777, 333))]:
        a = torch.randn(na, device="cuda")
        b = torch.randn(nb, device="cuda")
        c_t = concat(a, b)
        c_r = torch.cat([a, b])
        max_diff = (c_t - c_r).abs().max().item()
        print(f"  [{name}] a={na} b={nb}  max_diff={max_diff:.2e}  "
              f"{'✅' if max_diff < 1e-7 else '❌'}")

    print("\n--- Performance ---")
    a = torch.randn(8388608, device="cuda", dtype=torch.float32)
    b = torch.randn(8388608, device="cuda", dtype=torch.float32)
    n = a.numel() + b.numel()
    result = bench_compare(
        {
            "Triton concat": lambda: concat(a, b),
            "PyTorch cat": lambda: torch.cat([a, b]),
        },
        flops=0,           # 纯数据搬运
        bytes_accessed=n * 4 * 3,  # read a+b, write c
        dtype="fp32",
    )
    print_compare_report(result)


# PERFORMANCE NOTES
# =================
# - Concat 是纯 memory-bound: 读两次 (a + b), 写一次 (output).
# - "Conditional" load: 两种 load 都会执行 (无 true branching),
#   因为 GPU 的 SIMT 模型: 一个 warp 内各 thread 可以有不同的 mask.
# - mask & in_a: 不在 a 范围的 thread 的 load 会被 predicate 掉.
# - 如果 BLOCK 正好跨 a/b 边界, 一半 thread 读 a 一半读 b → 正常并行.
# - 和 split (后续 kernel) 互为逆操作.

if __name__ == "__main__":
    main()
