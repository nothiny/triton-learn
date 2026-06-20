"""
41_max_pool1d.py — 1D Max Pooling with Stride

学习目标:
  - 掌握 sliding window + reduction 的池化模式
  - 理解 stride (步长) 对 grid 和访存模式的影响
  - 学习 pool 的 "window over continuous data" 实现

数学定义:
  output[i] = max(x[i*stride : i*stride + kernel_size])

为什么 GPU 上 Pooling 和 CPU 完全不同:
  - CPU: O(N*kernel) sequential scan
  - GPU: O(N/stride) programs, 每个独立读完整个 window
  - 每个 window 有重叠, 但 GPU 上重复读比共享中间结果更简单

应用:
  - CNN downsampling (经典 AlexNet/VGG)
  - 虽然现在流行 stride-2 conv 替代 pooling,
    但 max_pool1d 仍是时序模型 (WaveNet, TCN) 的基础

运行: python phase1_fundamentals/41_max_pool1d.py
"""

import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch, triton, triton.language as tl
from utils.profiler import bench_compare, print_compare_report


@triton.jit
def max_pool1d_kernel(x_ptr, output_ptr, n_elements,
                       kernel_size, stride,
                       BLOCK_SIZE: tl.constexpr):
    """
    1D max pooling: output[i] = max(x[i*stride : i*stride + kernel_size]).
    每个 program 处理一批输出位置.
    """
    pid = tl.program_id(0)
    # 本 block 处理的输出位置范围
    out_start = pid * BLOCK_SIZE
    out_offsets = out_start + tl.arange(0, BLOCK_SIZE)

    # 各输出位置对应的输入窗口起始位置
    in_starts = out_offsets * stride
    in_mask = in_starts + kernel_size - 1 < n_elements

    # [GPU] 每个 thread 遍历自己的 window, 找 max
    # 用 tl.maximum 累积, O(kernel_size) 次迭代
    result = tl.full([BLOCK_SIZE], float("-inf"), dtype=tl.float32)
    for k in range(0, kernel_size):
        in_offsets = in_starts + k
        k_mask = in_mask & (in_offsets < n_elements)
        val = tl.load(x_ptr + in_offsets, mask=k_mask, other=float("-inf"))
        result = tl.maximum(result, val)

    tl.store(output_ptr + out_offsets, result, mask=in_mask)


def max_pool1d(x: torch.Tensor, kernel_size: int, stride: int = None
               ) -> torch.Tensor:
    """1D max pooling. """
    if stride is None:
        stride = kernel_size
    n = x.numel()
    n_out = (n - kernel_size) // stride + 1
    output = torch.empty(n_out, device=x.device, dtype=torch.float32)
    BLOCK_SIZE = 128
    grid = (triton.cdiv(n_out, BLOCK_SIZE),)
    max_pool1d_kernel[grid](x, output, n, kernel_size, stride,
                             BLOCK_SIZE=BLOCK_SIZE)
    return output


def main():
    print("=" * 60)
    print("41_max_pool1d — 1D Max Pooling")
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.manual_seed(42)

    for name, (n, ks, st) in [("small ", (64, 3, 2)), ("medium", (256, 5, 2)),
                                ("large ", (1024, 7, 4))]:
        x = torch.randn(n, device="cuda")
        y_t = max_pool1d(x, ks, st)
        y_r = torch.nn.functional.max_pool1d(
            x.view(1, 1, n), ks, stride=st, padding=0,
        ).flatten()
        max_diff = (y_t - y_r).abs().max().item()
        print(f"  [{name}] n={n} k={ks} s={st}  max_diff={max_diff:.2e}  "
              f"{'✅' if max_diff < 1e-6 else '❌'}")

    print("\n--- Performance ---")
    x = torch.randn(1048576, device="cuda", dtype=torch.float32)
    n = x.numel()
    n_out = (n - 3) // 2 + 1
    result = bench_compare(
        {
            "Triton max_pool1d": lambda: max_pool1d(x, 3, 2),
            "PyTorch max_pool1d": lambda: torch.nn.functional.max_pool1d(
                x.view(1, 1, n), 3, stride=2).flatten(),
        },
        flops=n_out * 3 * 2,  # ~compare per window element
        bytes_accessed=n * 4 + n_out * 4,  # read all, write output
        dtype="fp32",
    )
    print_compare_report(result)


# PERFORMANCE NOTES
# =================
# - 每个 output 读 kernel_size 个 input → O(N * kernel_size) 读取量.
# - 相邻 window 有大量重叠 (strided kernel_size > stride 时),
#   但本实现不做优化 — 重复读比复杂的 shared memory 窗口共享更简单.
# - Memory-bound (除非 kernel_size >> stride, 此时 compute 增多).
# - 二维/三维 max pool 可以类似扩展: 用更高维 grid.

if __name__ == "__main__":
    main()
