"""
03_fused_relu_bias.py — Fused ReLU + Bias kernel

学习目标：
  - 理解 operator fusion 为何重要
  - 学习 Triton 中 elementwise 操作的组合
  - 对比 fused vs sequential kernel 调用的性能差异

运行: python phase1_fundamentals/03_fused_relu_bias.py
"""

import torch
import triton
import triton.language as tl


@triton.jit
def relu_bias_kernel(
    x_ptr,      # 输入
    bias_ptr,   # bias 向量 (可广播)
    output_ptr, # 输出
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    单 kernel 完成: output = ReLU(x + bias)

    优势：x + bias 的结果不写回 HBM，直接在寄存器中传给 ReLU。
    如果拆成两个 kernel: x + bias → write HBM → load HBM → ReLU
      每个 element 多了一次 read + write = 2x 带宽消耗
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # 加载输入
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # bias 广播: 对于 2D tensor (M, N)，bias 沿行广播
    # 简化实现: bias 视为标量广播
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)

    # Fused: x + bias → ReLU，全程在寄存器中
    # ReLU(x) = max(0, x)
    result = tl.maximum(x + bias, 0.0)

    tl.store(output_ptr + offsets, result, mask=mask)


def fused_relu_bias(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Fused ReLU(x + bias)"""
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    relu_bias_kernel[grid](x, bias, output, n_elements, BLOCK_SIZE=1024)
    return output


def main():
    print("=" * 60)
    print("03_fused_relu_bias — Operator Fusion")
    print("=" * 60)

    size = 16777216  # 16M elements
    x = torch.randn(size, device="cuda")
    bias = torch.randn(size, device="cuda")

    # 正确性
    out_triton = fused_relu_bias(x, bias)
    out_torch = torch.relu(x + bias)

    max_diff = (out_triton - out_torch).abs().max().item()
    print(f"  Max diff: {max_diff:.6e}  {'✅' if max_diff < 1e-5 else '❌'}")

    # 性能对比: fused vs sequential
    n_iter = 100

    # Fused kernel
    for _ in range(10):
        _ = fused_relu_bias(x, bias)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(n_iter):
        _ = fused_relu_bias(x, bias)
    end.record()
    torch.cuda.synchronize()
    fused_ms = start.elapsed_time(end) / n_iter

    # Sequential: x + bias → write → read → ReLU
    start.record()
    for _ in range(n_iter):
        _ = torch.relu(x + bias)
    end.record()
    torch.cuda.synchronize()
    torch_ms = start.elapsed_time(end) / n_iter

    print(f"\n  Fused Triton:   {fused_ms:.4f} ms")
    print(f"  PyTorch fused:  {torch_ms:.4f} ms")


# PERFORMANCE NOTES
# =================
# - Operator fusion 对 memory-bound 操作特别有效（减少 round-trip）
# - [COMPILER] Triton 内部也有 TritonCombineOps pass，会自动融合相邻的
#   elementwise op，但显式写在 kernel 中更可控
# - 实际加速比取决于算术强度: 越 memory-bound → fusion 收益越大


if __name__ == "__main__":
    main()
