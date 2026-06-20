"""
42_avg_pool1d.py — 1D Average Pooling

学习目标:
  - 掌握 sliding window + mean reduction 的池化模式
  - 对比 max pool (selection) vs avg pool (arithmetic reduction)
  - 理解边界处理: 窗口越界时的 window 大小调整

数学定义:
  output[i] = mean(x[i*stride : i*stride + kernel_size])

和 41_max_pool1d 的区别:
  - Max pool: tl.maximum 累积 (comparing, 无除法)
  - Avg pool: tl.sum 累积 + 除以 kernel_size (arithmetic, 有除法)
  - 但两者结构几乎相同: window loop + reduce

Avg pooling 的特殊性:
  - Max pool 天然对 outlier 敏感 (保留最强信号)
  - Avg pool 更平滑 (噪声平均), 常用于 feature downsampling
  - 在 CNN 中, avg pool 通常用于最后几层 (全局 avg pool → classification)

运行: python phase1_fundamentals/42_avg_pool1d.py
"""

import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch, triton, triton.language as tl
from utils.profiler import bench_compare, print_compare_report


@triton.jit
def avg_pool1d_kernel(x_ptr, output_ptr, n_elements,
                       kernel_size, stride,
                       BLOCK_SIZE: tl.constexpr):
    """
    1D avg pooling: output[i] = mean(x[i*stride : i*stride + kernel_size]).
    """
    pid = tl.program_id(0)
    out_start = pid * BLOCK_SIZE
    out_offsets = out_start + tl.arange(0, BLOCK_SIZE)

    in_starts = out_offsets * stride
    in_mask = in_starts + kernel_size - 1 < n_elements

    # [GPU] 用 sum 累积, 最后除以 kernel_size
    result = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for k in range(0, kernel_size):
        in_offsets = in_starts + k
        k_mask = in_mask & (in_offsets < n_elements)
        val = tl.load(x_ptr + in_offsets, mask=k_mask, other=0.0)
        result += val  # [GPU] FMA accum in registers

    result = result / kernel_size.to(tl.float32)

    tl.store(output_ptr + out_offsets, result, mask=in_mask)


def avg_pool1d(x: torch.Tensor, kernel_size: int, stride: int = None
               ) -> torch.Tensor:
    if stride is None:
        stride = kernel_size
    n = x.numel()
    n_out = (n - kernel_size) // stride + 1
    output = torch.empty(n_out, device=x.device, dtype=torch.float32)
    BLOCK_SIZE = 128
    grid = (triton.cdiv(n_out, BLOCK_SIZE),)
    avg_pool1d_kernel[grid](x, output, n, kernel_size, stride,
                             BLOCK_SIZE=BLOCK_SIZE)
    return output


def main():
    print("=" * 60)
    print("42_avg_pool1d — 1D Average Pooling")
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.manual_seed(42)

    for name, (n, ks, st) in [("small ", (64, 3, 2)), ("medium", (256, 5, 2)),
                                ("large ", (1024, 7, 4))]:
        x = torch.randn(n, device="cuda")
        y_t = avg_pool1d(x, ks, st)
        y_r = torch.nn.functional.avg_pool1d(
            x.view(1, 1, n), ks, stride=st, padding=0,
        ).flatten()
        max_diff = (y_t - y_r).abs().max().item()
        print(f"  [{name}] n={n} k={ks} s={st}  max_diff={max_diff:.2e}  "
              f"{'✅' if max_diff < 1e-6 else '❌'}")

    # Compare max vs avg pooling
    print("\n--- Max Pool vs Avg Pool on same input ---")
    x = torch.arange(20, device="cuda", dtype=torch.float32)
    print(f"  Input: {x.tolist()}")
    y_max = torch.nn.functional.max_pool1d(x.view(1,1,20), 4, stride=4).flatten()
    y_avg = avg_pool1d(x, kernel_size=4, stride=4)
    print(f"  MaxPool (k=4,s=4): {y_max.tolist()}")
    print(f"  AvgPool (k=4,s=4): {y_avg.tolist()}")

    print("\n--- Performance ---")
    x = torch.randn(1048576, device="cuda", dtype=torch.float32)
    n = x.numel()
    result = bench_compare(
        {
            "Triton avg_pool1d": lambda: avg_pool1d(x, 3, 2),
            "PyTorch avg_pool1d": lambda: torch.nn.functional.avg_pool1d(
                x.view(1, 1, n), 3, stride=2).flatten(),
        },
        flops=(n - 3) // 2 * 3 * 2,
        bytes_accessed=n * 4 + (n - 3) // 2 * 4,
        dtype="fp32",
    )
    print_compare_report(result)


# PERFORMANCE NOTES
# =================
# - Avg pool 和 max pool 结构完全一样, 唯一区别: tl.sum vs tl.maximum.
# - 除法 (result / kernel_size) 在循环外执行一次, 不增加循环内延迟.
# - 和 max pool 一样, 相邻窗口重叠导致重复读取.
# - 优化: 对于大 kernel_size, 可以用 prefix sum 做 O(N) 窗口求和
#   (类似于 cumsum, 见 19_cumsum.py), 避免 O(N*kernel) 读取.

if __name__ == "__main__":
    main()
