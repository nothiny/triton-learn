"""
01_vector_add.py — 最基础的 Triton kernel

学习目标：
  - 理解 @triton.jit 装饰器
  - 掌握 tl.program_id, tl.arange, tl.load, tl.store
  - 了解 BLOCK_SIZE 对 occupancy 的影响
  - 初步使用 @triton.autotune

运行: python phase1_fundamentals/01_vector_add.py
"""

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernel 定义
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=[
        # BLOCK_SIZE 必须是 2 的幂，受 shared memory 和寄存器限制
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    ],
    key=["n_elements"],  # 根据元素数量选择最优配置
)
@triton.jit
def vector_add_kernel(
    x_ptr,  # 输入指针 A
    y_ptr,  # 输入指针 B
    out_ptr,  # 输出指针
    n_elements,  # 总元素数
    BLOCK_SIZE: tl.constexpr,  # [COMPILER] 编译时常量，每个 program 处理 BLOCK_SIZE 个元素
):
    """
    向量加法: out[i] = x[i] + y[i]

    每个 program (等同于 CUDA thread block) 处理 BLOCK_SIZE 个元素。
    GPU 上会有 n_elements / BLOCK_SIZE 个 program 并行执行。
    """
    # program_id(0) = blockIdx.x，全局 block 索引
    pid = tl.program_id(axis=0)

    # 当前 program 负责的元素范围: [block_start, block_start + BLOCK_SIZE)
    block_start = pid * BLOCK_SIZE

    # tl.arange 生成 [0, 1, 2, ..., BLOCK_SIZE-1]
    # [COMPILER] 这会在编译时展开为常量向量
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # mask: 处理 n_elements 不是 BLOCK_SIZE 整数倍的情况
    # 最后一个 block 的尾部元素需要 mask 掉
    mask = offsets < n_elements

    # 从 HBM 加载数据（编译器自动处理 coalescing）
    # other=0.0: 越界元素的默认值
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # 向量加法 — 逐元素并行
    output = x + y

    # 写回 HBM
    tl.store(out_ptr + offsets, output, mask=mask)


# ---------------------------------------------------------------------------
# 包装函数
# ---------------------------------------------------------------------------


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """调用 Triton kernel 的 Python 包装"""
    output = torch.empty_like(x)
    n_elements = x.numel()

    # grid: 每个 program 处理 BLOCK_SIZE 个元素
    # triton.cdiv = ceil division
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    vector_add_kernel[grid](x, y, output, n_elements)
    return output


# ---------------------------------------------------------------------------
# 测试 & 性能对比
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("01_vector_add — Triton vs PyTorch")
    print("=" * 60)

    # GPU 信息
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
    print()

    # 测试正确性
    sizes = [1024, 65536, 1048576, 16777216]  # 1K → 16M elements
    for size in sizes:
        x = torch.rand(size, device="cuda", dtype=torch.float32)
        y = torch.rand(size, device="cuda", dtype=torch.float32)

        out_triton = vector_add(x, y)
        out_torch = x + y

        max_diff = (out_triton - out_torch).abs().max().item()
        status = "✅" if max_diff < 1e-5 else "❌"
        print(f"  size={size:>10,}  max_diff={max_diff:.2e}  {status}")

    # 性能对比
    print("\n--- Performance (16M elements) ---")
    size = 16777216
    x = torch.rand(size, device="cuda", dtype=torch.float32)
    y = torch.rand(size, device="cuda", dtype=torch.float32)

    # Warmup
    for _ in range(10):
        _ = vector_add(x, y)
    torch.cuda.synchronize()

    # Benchmark Triton
    import time

    n_iter = 100
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        _ = vector_add(x, y)
    end.record()
    torch.cuda.synchronize()
    triton_ms = start.elapsed_time(end) / n_iter

    # Benchmark PyTorch
    start.record()
    for _ in range(n_iter):
        _ = x + y
    end.record()
    torch.cuda.synchronize()
    torch_ms = start.elapsed_time(end) / n_iter

    # 带宽计算: read 2×4bytes + write 1×4bytes = 12 bytes/element
    bytes_total = size * 3 * 4  # x(4B) + y(4B) + out(4B)
    bw_triton = bytes_total / (triton_ms * 1e-3) / 1e9  # GB/s
    bw_torch = bytes_total / (torch_ms * 1e-3) / 1e9

    print(f"  Triton:  {triton_ms:.4f} ms  ({bw_triton:.1f} GB/s)")
    print(f"  PyTorch: {torch_ms:.4f} ms  ({bw_torch:.1f} GB/s)")


# PERFORMANCE NOTES
# =================
# - 向量加是典型的 memory-bound 操作：算术强度 = (1 FLOP) / (12 bytes) = 0.083 FLOP/byte
# - H100 HBM 带宽 3.35 TB/s → 理论峰值 ~280 GB/s (实际 ~80%)
# - BLOCK_SIZE 选择指南:
#     - 太小 (<128): 每个 program 开销大，GPU 利用率低
#     - 太大 (>2048): 寄存器/spill 压力增大，occupancy 下降
#     - 最优: 256-1024，保证足够的并行 program 数量
# - num_warps: 更多 warp 有助于隐藏延迟，但每个 warp 的寄存器减少
#   [COMPILER] 类比: num_warps = 每个 program 的"硬件线程数"


if __name__ == "__main__":
    main()
