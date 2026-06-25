"""
21_online_softmax_backward.py — Online Softmax Backward Pass

学习目标:
  - 理解 softmax 反向传播的数学和数值实现
  - 掌握如何在一次 pass 中计算 dX（避免存储完整 softmax 矩阵）
  - 学会利用 forward 的 output 直接计算 backward

数学推导:
  Forward:  y_i = exp(x_i - max) / sum_j(exp(x_j - max))

  Backward: 已知 dy_i，求 dx_i
    dx_i = y_i * (dy_i - sum_j(y_j * dy_j))
         = y_i * dy_i - y_i * (y · dy)

  关键: 只需要 forward 的 output y 和 upstream gradient dy，
  不需要保存完整 softmax 矩阵。

运行: python phase2_compute/21_online_softmax_backward.py
"""

import torch
import triton
import triton.language as tl
from triton.testing import do_bench


@triton.jit
def softmax_fwd_kernel(
    x_ptr, output_ptr,
    N,
    stride_xn,
    BLOCK_N: tl.constexpr,
):
    """
    Forward: online softmax (2-pass)。

    Pass 1: 计算 running max m 和 running sum s
    Pass 2: 用 m 和 s 归一化并写回
    """
    row_idx = tl.program_id(axis=0)

    m = tl.full([1], float("-inf"), dtype=tl.float32)
    s = tl.zeros([1], dtype=tl.float32)

    num_tiles = tl.cdiv(N, BLOCK_N)
    for tile_idx in range(num_tiles):
        col_offs = tile_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        x_ptrs = x_ptr + row_idx * stride_xn + col_offs
        tile = tl.load(x_ptrs, mask=col_offs < N, other=float("-inf"))

        tile_m = tl.max(tile, axis=0)
        m_new = tl.maximum(m, tile_m)
        s = s * tl.exp(m - m_new) + tl.sum(tl.exp(tile - m_new), axis=0)
        m = m_new

    for tile_idx in range(num_tiles):
        col_offs = tile_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        x_ptrs = x_ptr + row_idx * stride_xn + col_offs
        tile = tl.load(x_ptrs, mask=col_offs < N, other=float("-inf"))

        out_tile = tl.exp(tile - m) / s
        out_ptrs = output_ptr + row_idx * stride_xn + col_offs
        tl.store(out_ptrs, out_tile, mask=col_offs < N)


@triton.jit
def softmax_bwd_kernel(
    output_ptr,      # y = softmax(x) from forward (M, N)
    dy_ptr,          # (M, N) — upstream gradient
    dx_ptr,          # (M, N) — output gradient
    N,
    stride_yn, stride_dyn, stride_dxn,
    BLOCK_N: tl.constexpr,
):
    """
    Backward: dx_i = y_i * (dy_i - dot(y, dy))

    两 pass:
      Pass 1: 计算 s = sum_j(y_j * dy_j)
      Pass 2: dx_i = y_i * (dy_i - s)
    """
    row_idx = tl.program_id(axis=0)

    # Pass 1: dot product s = sum(y_i * dy_i)
    s = tl.zeros([1], dtype=tl.float32)
    num_tiles = tl.cdiv(N, BLOCK_N)
    for tile_idx in range(num_tiles):
        col_offs = tile_idx * BLOCK_N + tl.arange(0, BLOCK_N)

        y = tl.load(output_ptr + row_idx * stride_yn + col_offs,
                    mask=col_offs < N, other=0.0)
        dy = tl.load(dy_ptr + row_idx * stride_dyn + col_offs,
                     mask=col_offs < N, other=0.0)
        s += tl.sum(y * dy, axis=0)

    # Pass 2: dx = y * (dy - s)
    for tile_idx in range(num_tiles):
        col_offs = tile_idx * BLOCK_N + tl.arange(0, BLOCK_N)

        y = tl.load(output_ptr + row_idx * stride_yn + col_offs,
                    mask=col_offs < N, other=0.0)
        dy = tl.load(dy_ptr + row_idx * stride_dyn + col_offs,
                     mask=col_offs < N, other=0.0)

        dx = y * (dy - s)
        tl.store(dx_ptr + row_idx * stride_dxn + col_offs,
                 dx, mask=col_offs < N)


def softmax_fwd(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    output = torch.empty_like(x)
    grid = (M,)
    softmax_fwd_kernel[grid](
        x, output, N, x.stride(0), BLOCK_N=128,
    )
    return output


def softmax_bwd(output: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    M, N = output.shape
    dx = torch.empty_like(output)
    grid = (M,)
    softmax_bwd_kernel[grid](
        output, dy, dx,
        N,
        output.stride(0), dy.stride(0), dx.stride(0),
        BLOCK_N=128,
    )
    return dx


def main():
    print("=" * 60)
    print("21_online_softmax_backward — Softmax Forward + Backward")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    for M, N in [(64, 128), (128, 256), (256, 512), (512, 1024)]:
        x = torch.randn(M, N, device="cuda", dtype=torch.float32)

        # Triton forward + backward
        y_triton = softmax_fwd(x)

        # PyTorch reference forward
        y_ref = torch.softmax(x, dim=-1)
        fwd_diff = (y_triton - y_ref).abs().max().item()

        # Backward
        dy = torch.randn_like(y_triton)
        dx_triton = softmax_bwd(y_triton, dy)

        # PyTorch reference backward (use x with grad)
        x_ref = x.clone().requires_grad_(True)
        y_ref = torch.softmax(x_ref, dim=-1)
        y_ref.backward(dy)
        dx_ref = x_ref.grad
        bwd_diff = (dx_triton - dx_ref).abs().max().item()

        ms_fwd = do_bench(lambda: softmax_fwd(x))
        ms_bwd = do_bench(lambda: softmax_bwd(y_triton, dy))

        status = "✅" if fwd_diff < 1e-4 and bwd_diff < 1e-4 else "❌"
        print(f"  {M}×{N}: fwd={ms_fwd:.4f}ms  bwd={ms_bwd:.4f}ms  "
              f"fwd_diff={fwd_diff:.2e}  bwd_diff={bwd_diff:.2e}  {status}")


# PERFORMANCE NOTES
# =================
# - Softmax backward 使用 forward 的 output y 直接计算 dx = y * (dy - y·dy)
# - 在 Flash Attention backward 中还需要 recompute softmax（因为不存 attention matrix）
# - Backward 也是 memory-bound（读 y + dy，写 dx）
# - Online softmax forward 是 2-pass（读 2 次 + 写 1 次）
# - Backward 也是 2-pass（读 y+dy 各 2 次 + 写 1 次）


if __name__ == "__main__":
    main()
