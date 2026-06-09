"""
06_depthwise_conv.py — Depthwise Convolution in Triton

Depthwise conv: 每个输入 channel 有独立的 filter kernel。
对于 (N, C, H, W) 输入和 (C, 1, KH, KW) filter (groups=C):
  output[n, c, h, w] = sum_{i=0}^{KH-1} sum_{j=0}^{KW-1}
                         input[n, c, h+i, w+j] * filter[c, 0, i, j]

与标准卷积的区别:
  - 标准 conv: C_out × C_in × KH × KW 参数，不同 channel 之间全连接
  - Depthwise:   C_in × 1 × KH × KW 参数，每个 channel 独立的 filter
  - MobileNet / EfficientNet 的基础 building block

与 GEMM 的关系:
  - Depthwise conv 可以通过 im2col → GEMM 来实现
  - 但直接实现避免了 im2col 的内存开销
  - 算术强度比 GEMM 低得多（每个输出元素只做 KH*KW 次乘加）
  - 所以通常是 memory-bound

运行: python phase2_compute/06_depthwise_conv.py
"""

import torch
import triton
import triton.language as tl


@triton.jit
def depthwise_conv_kernel(
    x_ptr,          # 输入: (N, C, H, W)
    w_ptr,          # filter: (C, 1, KH, KW)
    y_ptr,          # 输出: (N, C, H_out, W_out)
    N, C, H, W,
    KH, KW,         # filter 大小
    stride_h, stride_w,
    pad_h, pad_w,
    H_out, W_out,
    # Strides
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wc, stride_wwh, stride_wkh, stride_wkw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """
    Depthwise convolution kernel.

    每个 program 处理一个 (batch, channel_group) 的输出 tile。

    策略: 沿着 H_out × W_out 做 tiling，在 KH × KW 维度做 reduction。
    """
    pid = tl.program_id(axis=0)
    num_c_blocks = tl.cdiv(C, BLOCK_C)
    num_h_blocks = tl.cdiv(H_out, BLOCK_H)
    num_w_blocks = tl.cdiv(W_out, BLOCK_W)
    num_blocks_per_batch = num_c_blocks * num_h_blocks * num_w_blocks

    batch_idx = pid // num_blocks_per_batch
    pid_rem = pid % num_blocks_per_batch

    c_block_idx = pid_rem // (num_h_blocks * num_w_blocks)
    pid_rem2 = pid_rem % (num_h_blocks * num_w_blocks)
    h_block_idx = pid_rem2 // num_w_blocks
    w_block_idx = pid_rem2 % num_w_blocks

    # Output spatial ranges
    offs_h = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)  # [BLOCK_H]
    offs_w = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)  # [BLOCK_W]
    offs_c = c_block_idx * BLOCK_C + tl.arange(0, BLOCK_C)  # [BLOCK_C]

    # Input spatial ranges (accounting for stride and padding)
    in_h = offs_h[:, None] * stride_h + tl.arange(0, KH)[None, :] - pad_h  # [BLOCK_H, KH]
    in_w = offs_w[:, None] * stride_w + tl.arange(0, KW)[None, :] - pad_w  # [BLOCK_W, KW]

    # Masks
    h_mask = (offs_h[:, None] < H_out)  # [BLOCK_H, 1]
    w_mask = (offs_w[:, None] < W_out)  # [BLOCK_W, 1]
    c_mask = (offs_c[:, None, None] < C)  # [BLOCK_C, 1, 1]

    # Accumulator: [BLOCK_C, BLOCK_H, BLOCK_W]
    acc = tl.zeros([BLOCK_C, BLOCK_H, BLOCK_W], dtype=tl.float32)

    # Reduction over KH × KW (filter spatial dims)
    for kh_idx in range(KH):
        for kw_idx in range(KW):
            # Input position for this filter element
            cur_h = in_h[:, kh_idx]   # [BLOCK_H]
            cur_w = in_w[:, kw_idx]   # [BLOCK_W]

            # Validity mask for input access
            valid_h = (cur_h >= 0) & (cur_h < H)  # [BLOCK_H]
            valid_w = (cur_w >= 0) & (cur_w < W)  # [BLOCK_W]
            valid = valid_h[None, :, None] & valid_w[None, None, :]  # [1, BLOCK_H, BLOCK_W]

            # Load input tile: [BLOCK_C, BLOCK_H, BLOCK_W] → load [BLOCK_C, BLOCK_H] before broadcast
            # Actually we need to gather along spatial dims
            # Use x_ptr + batch_idx*stride_xn + c*stride_xc + h*stride_xh + w*stride_xw
            for c_local in range(BLOCK_C):
                c_global = offs_c[c_local]
                if c_global >= C:
                    continue

                x_ptrs = (x_ptr + batch_idx * stride_xn +
                          c_global * stride_xc +
                          cur_h[None, :] * stride_xh +
                          cur_w[None, :] * stride_xw)  # [BLOCK_H, BLOCK_W]
                x = tl.load(x_ptrs, mask=valid, other=0.0)

                # Load filter weight for this (c, kh, kw)
                w_val = tl.load(w_ptr + c_global * stride_wc +
                                kh_idx * stride_wkh + kw_idx * stride_wkw)

                # Accumulate
                acc = tl.where(
                    (tl.arange(0, BLOCK_C)[:, None, None] == c_local),
                    acc + x[None, :, :] * w_val,
                    acc,
                )

    # Simplified acc (avoiding complex indexing):
    # Use a flatter approach — write per-channel
    acc_out = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)

    # Store output
    out_h_mask = offs_h[:, None] < H_out  # [BLOCK_H, 1]
    out_w_mask = offs_w[None, :] < W_out  # [1, BLOCK_W]

    for c_local in range(BLOCK_C):
        c_global = offs_c[c_local]
        if c_global >= C:
            continue

        y_ptrs = (y_ptr + batch_idx * stride_yn +
                  c_global * stride_yc +
                  offs_h[:, None] * stride_yh +
                  offs_w[None, :] * stride_yw)  # [BLOCK_H, BLOCK_W]

        tl.store(y_ptrs, acc[c_local], mask=out_h_mask & out_w_mask)


def depthwise_conv(
    x: torch.Tensor,        # (N, C, H, W)
    weight: torch.Tensor,    # (C, 1, KH, KW)
    stride: int = 1,
    padding: int = 0,
) -> torch.Tensor:
    """
    Depthwise convolution: each input channel has its own filter.

    Args:
        x: Input tensor (N, C, H, W)
        weight: Filter (C, 1, KH, KW)
        stride: Spatial stride
        padding: Spatial padding

    Returns:
        Output tensor (N, C, H_out, W_out)
    """
    N, C, H, W = x.shape
    C_w, _, KH, KW = weight.shape
    assert C == C_w, f"Channel mismatch: input {C} vs weight {C_w}"

    H_out = (H + 2 * padding - KH) // stride + 1
    W_out = (W + 2 * padding - KW) // stride + 1

    y = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)

    BLOCK_H = 16
    BLOCK_W = 16
    BLOCK_C = 4  # process 4 channels per program

    grid = (N * triton.cdiv(C, BLOCK_C) *
            triton.cdiv(H_out, BLOCK_H) *
            triton.cdiv(W_out, BLOCK_W),)

    depthwise_conv_kernel[grid](
        x, weight, y,
        N, C, H, W,
        KH, KW,
        stride, stride,
        padding, padding,
        H_out, W_out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, BLOCK_C=BLOCK_C,
    )
    return y


def ref_depthwise_conv(x, weight, stride=1, padding=0):
    """PyTorch reference using F.conv2d with groups=C."""
    C = x.shape[1]
    return torch.nn.functional.conv2d(
        x, weight, stride=stride, padding=padding, groups=C
    )


def main():
    print("=" * 60)
    print("06_depthwise_conv — Triton vs PyTorch")
    print("=" * 60)

    # Standard depthwise conv: MobileNet-style
    N, C, H, W = 4, 32, 56, 56
    KH, KW = 3, 3
    stride, padding = 1, 1

    torch.manual_seed(42)
    x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float32)
    weight = torch.randn(C, 1, KH, KW, device="cuda", dtype=torch.float32)

    # Correctness
    y_triton = depthwise_conv(x, weight, stride=stride, padding=padding)
    y_ref = ref_depthwise_conv(x, weight, stride=stride, padding=padding)

    max_diff = (y_triton - y_ref).abs().max().item()
    status = "✅" if max_diff < 1e-3 else "❌"
    print(f"  Shape: ({N}, {C}, {H}, {W}) → ({N}, {C}, "
          f"{y_triton.shape[2]}, {y_triton.shape[3]})")
    print(f"  Filter: ({C}, 1, {KH}, {KW})")
    print(f"  Max diff: {max_diff:.6e}  {status}")

    # Performance
    n_iter = 100
    for _ in range(10):
        depthwise_conv(x, weight, stride=stride, padding=padding)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(n_iter):
        depthwise_conv(x, weight, stride=stride, padding=padding)
    end.record()
    torch.cuda.synchronize()
    triton_ms = start.elapsed_time(end) / n_iter

    start.record()
    for _ in range(n_iter):
        ref_depthwise_conv(x, weight, stride=stride, padding=padding)
    end.record()
    torch.cuda.synchronize()
    torch_ms = start.elapsed_time(end) / n_iter

    # Memory bandwidth analysis
    in_bytes = x.numel() * x.element_size()
    w_bytes = weight.numel() * weight.element_size()
    out_bytes = y_triton.numel() * y_triton.element_size()
    total_bytes = in_bytes + w_bytes + out_bytes

    triton_bw = total_bytes / (triton_ms * 1e-3) / 1e9
    torch_bw = total_bytes / (torch_ms * 1e-3) / 1e9

    print(f"\n  Triton:   {triton_ms:.4f} ms  ({triton_bw:.1f} GB/s)")
    print(f"  PyTorch:  {torch_ms:.4f} ms  ({torch_bw:.1f} GB/s)")


# PERFORMANCE NOTES
# =================
# - Depthwise conv 是典型的 memory-bound kernel:
#   - 算术强度: KH*KW FLOP / (4 + KH*KW) bytes ≈ 9/13 ≈ 0.7 FLOP/byte (KH=KW=3)
#   - 远低于 H100 ridge point (295 FLOP/byte)
# - 主要优化方向:
#   1. 增加 BLOCK_C — 多个 channel 共享 filter 数据
#   2. 与 pointwise conv (1×1) 融合 — 减少一次 HBM round-trip
#   3. 使用 shared memory 缓存 filter（weight stationary）
# - cuDNN 的 depthwise conv 实现使用了隐式 GEMM（im2col → matrix multiply）
#   这在 large KH/KW 下可能更快，但 im2col 本身有内存开销
# - [COMPILER] Triton 将嵌套的 for kh/kw 循环完全展开（因为 KH, KW 通常是小的编译时常量）


if __name__ == "__main__":
    main()
