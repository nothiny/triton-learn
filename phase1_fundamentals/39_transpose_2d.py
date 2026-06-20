"""
39_transpose_2d.py — 2D Transpose & Coalesced Access Analysis

学习目标:
  - 掌握 2D transpose 的 GPU 实现
  - 深入理解 coalesced (合并) vs strided (跨步) memory access
  - 学习 shared memory 解决 strided 访存的经典模式

Memory Access 分析:
  读 M×N 矩阵, 写 N×M:
  ┌────────────────┬─────────────┬─────────────┐
  │ 访问模式        │ 读 (src)     │ 写 (dst)     │
  ├────────────────┼─────────────┼─────────────┤
  │ Naive            │ coalesced   │ strided     │
  │ 通过 shared mem  │ coalesced   │ coalesced   │
  └────────────────┴─────────────┴─────────────┘

Coalesced Access 为什么重要:
  - 一次 warp 指令 (32 threads) 读连续 128 bytes → 1 L1 transaction
  - Strided: 每个 thread 的数据不在同一 cache line → 32 transactions
  - 后者有效带宽差 10-100×

运行: python phase1_fundamentals/39_transpose_2d.py
"""

import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch, triton, triton.language as tl
from utils.profiler import bench_compare, print_compare_report


@triton.jit
def transpose_2d_kernel(x_ptr, output_ptr, M, N,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """
    2D transpose: out[j, i] = x[i, j].
    用 2D grid, 每个 program 转置一个 BLOCK_M × BLOCK_N 的 tile.

    [GPU] 读: coalesced (连续行), 写: strided (列→行, 需要跨步).
          可以通过 shared memory 中转来让写也 coalesced,
          但这里展示的是直接 strided write (简单且通常足够快).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 源矩阵的 tile 位置
    row_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    row_mask = row_offs[:, None] < M   # [BLOCK_M, 1]
    col_mask = col_offs[None, :] < N   # [1, BLOCK_N]
    mask = row_mask & col_mask          # [BLOCK_M, BLOCK_N]

    # [GPU] 读: coalesced (BLOCK_M 行, 每行 BLOCK_N 连续列)
    src_offs = row_offs[:, None] * N + col_offs[None, :]
    x_tile = tl.load(x_ptr + src_offs, mask=mask, other=0.0)

    # [GPU] 写: strided (每个 thread 在 dst 中跨步 M)
    # dst[j][i] = src[i][j] → dst_offs = col * M + row
    dst_offs = col_offs[None, :] * M + row_offs[:, None]
    tl.store(output_ptr + dst_offs, x_tile, mask=mask)


def transpose_2d(x: torch.Tensor) -> torch.Tensor:
    """2D transpose via Triton kernel."""
    assert x.dim() == 2
    M, N = x.shape
    output = torch.empty(N, M, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),
                          triton.cdiv(N, meta["BLOCK_N"]))
    transpose_2d_kernel[grid](x, output, M, N,
                               BLOCK_M=32, BLOCK_N=32)
    return output


def main():
    print("=" * 60)
    print("39_transpose_2d — 2D Transpose & Coalescing Analysis")
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.manual_seed(42)

    for name, shape in [("small  ", (256, 128)), ("medium ", (1024, 512)),
                          ("square ", (2048, 2048))]:
        M, N = shape
        x = torch.randn(M, N, device="cuda")
        y_t = transpose_2d(x)
        y_r = x.T.contiguous()
        max_diff = (y_t - y_r).abs().max().item()
        ok = "✅" if max_diff < 1e-6 else "❌"
        print(f"  [{name}] shape=({M},{N}) "
              f"max_diff={max_diff:.2e}  {ok}")

    # Coalescing demo: compare read-coalesced vs write-coalesced timing
    print("\n--- Coalescing Analysis ---")
    M, N = 4096, 4096
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    import time
    # Warmup
    for _ in range(10): transpose_2d(x); x.T.contiguous()
    torch.cuda.synchronize()
    # Triton transpose
    t0 = time.perf_counter()
    for _ in range(50): transpose_2d(x)
    torch.cuda.synchronize()
    t_triton = (time.perf_counter() - t0) / 50 * 1000
    # PyTorch transpose
    t0 = time.perf_counter()
    for _ in range(50): x.T.contiguous()
    torch.cuda.synchronize()
    t_pt = (time.perf_counter() - t0) / 50 * 1000
    print(f"  Triton transpose: {t_triton:.4f} ms")
    print(f"  PyTorch transpose: {t_pt:.4f} ms")
    print(f"  Speedup: {t_pt / t_triton:.2f}x")
    print(f"\n  💡 Triton 读是 coalesced (行连续), 写是 strided (跨 M=4096).")
    print(f"     对转置来说, 总有一个方向是 strided — 无法同时 coalesced 读和写.")
    print(f"     生产级实现用 shared memory: 先 coalesced 读入 shared,")
    print(f"     再 coalesced 写出 (交换 shared mem 中的坐标).")

    print("\n--- Performance vs PyTorch ---")
    result = bench_compare(
        {
            "Triton transpose (ours)": lambda: transpose_2d(x),
            "PyTorch x.T.contiguous()": lambda: x.T.contiguous(),
        },
        flops=0,                # 纯数据搬运, 无计算
        bytes_accessed=M * N * 8,  # read + write
        dtype="fp32",
    )
    print_compare_report(result)


# PERFORMANCE NOTES
# =================
# - 2D transpose 是纯数据搬运, 无 FLOPs.
# - 瓶颈: strided access 导致 L1/L2 cache 利用率低.
# - Shared memory 方案: BLOCK_M×BLOCK_N tile 先 coalesced 读入 shared,
#   然后从 shared 交换坐标后再 coalesced 写出.
# - 对于 square matrix, bandwidth utilization ~40-60% (取决于 tile 大小).
# - 和 matmul 的关系: matmul 也需要转置 B (因为 B 按行存储但需要按列读取).

if __name__ == "__main__":
    main()
