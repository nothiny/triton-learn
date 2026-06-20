"""
50_fused_bias_gelu.py — Fused Bias + GELU Activation

学习目标:
  - 掌握 "bias add + activation" 的融合模式 (transformer FFN 关键优化)
  - 理解为什么 3-op fusion (load, bias add, activate, store) 比分别 launch 更好
  - 学习 elementwise fusion 的极限: 何时不值得继续融合

数学定义:
  output = GELU(x + bias)

为什么需要 fuse:
  - PyTorch non-fused: 3 次 kernel launch:
    1. x + bias → temp1       (read x, bias; write temp1)
    2. GELU(temp1) → output   (read temp1; write output)
    3. 总共: 4 reads + 3 writes to HBM
  - Triton fused: 1 次 kernel launch:
    1. 在寄存器中完成: load x, load bias, add, gelu, store
    2. 总共: 2 reads + 1 write to HBM
    3. 节省 50% HBM traffic

Transformer FFN 中的应用:
  FFN(x) = Linear2(Dropout(GELU(Linear1(x) + bias1))) + bias2
  其中 GELU(x+bias) 是最常见的融合点.

融合的极限:
  - 不是越多越好: 寄存器压力 (register pressure) 随融合数增加
  - 通常 3-5 个 ops 一次融合最佳 (更多 ops → 寄存器溢出 → spill to HBM)

运行: python phase1_fundamentals/50_fused_bias_gelu.py
"""

import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch, triton, triton.language as tl
from utils.profiler import bench_compare, print_compare_report


@triton.jit
def fused_bias_gelu_kernel(x_ptr, bias_ptr, output_ptr, n_elements, n_cols,
                            BLOCK_SIZE: tl.constexpr):
    """
    output = GELU(x + bias).
    x: (M, N) 或 (N,)
    bias: (N,) — broadcast 到所有行
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # [GPU] 1 次 load x, 1 次 load bias (用取模实现 broadcast)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    bias_offsets = offsets % n_cols  # [GPU] 取模 → 广播 bias 到所有行
    bias = tl.load(bias_ptr + bias_offsets, mask=mask, other=0.0)

    # bias add
    z = x + bias

    # [GPU] GELU ≈ z * sigmoid(2*inner) via MUFU (~16 cycles)
    # tanh(z)=2*sigmoid(2z)-1, GELU=z*0.5*(1+tanh(inner))=z*sigmoid(2*inner)
    inner = 0.7978845608028654 * (z + 0.044715 * z * z * z)
    result = z * tl.sigmoid(2.0 * inner)

    tl.store(output_ptr + offsets, result, mask=mask)


def fused_bias_gelu(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Fused bias + GELU: output = GELU(x + bias)."""
    if x.dim() == 2:
        assert bias.dim() == 1 and x.shape[-1] == bias.shape[0]
    elif x.dim() == 1:
        assert x.shape == bias.shape
    n = x.numel()
    n_cols = bias.numel()
    output = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    fused_bias_gelu_kernel[grid](x, bias, output, n, n_cols, BLOCK_SIZE=1024)
    return output


def main():
    print("=" * 60)
    print("50_fused_bias_gelu — Fused Bias + GELU Activation")
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.manual_seed(42)

    for name, shape in [("small ", (256, 128)), ("medium", (1024, 512)),
                          ("large ", (4096, 768))]:
        M, N = shape
        x = torch.randn(M, N, device="cuda")
        bias = torch.randn(N, device="cuda")
        y_t = fused_bias_gelu(x, bias)
        y_r = torch.nn.functional.gelu(x + bias, approximate="tanh")
        max_diff = (y_t - y_r).abs().max().item()
        print(f"  [{name}] shape=({M},{N})  max_diff={max_diff:.2e}  "
              f"{'✅' if max_diff < 1e-6 else '❌'}")

    # 1D case
    for name, size in [("1D small ", 256), ("1D medium", 65536)]:
        x = torch.randn(size, device="cuda")
        bias = torch.randn(size, device="cuda")
        y_t = fused_bias_gelu(x, bias)
        y_r = torch.nn.functional.gelu(x + bias, approximate="tanh")
        max_diff = (y_t - y_r).abs().max().item()
        print(f"  [{name}] size={size:8d}  max_diff={max_diff:.2e}  "
              f"{'✅' if max_diff < 1e-6 else '❌'}")

    # Fusion benefit demo
    print("\n--- Fusion Benefit (HBM Traffic Comparison) ---")
    M, N = 4096, 768
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    bias = torch.randn(N, device="cuda", dtype=torch.float32)
    n = x.numel()
    print(f"  Non-fused: 2× read x ({n*4/1e6:.1f} MB) + read bias "
          f"({N*4/1024:.1f} KB) + write temp ({n*4/1e6:.1f} MB) + "
          f"read temp ({n*4/1e6:.1f} MB) + write out = ~{n*4*4/1e6:.1f} MB")
    print(f"  Fused:     1× read x ({n*4/1e6:.1f} MB) + read bias "
          f"({N*4/1024:.1f} KB) + 1× write out ({n*4/1e6:.1f} MB) "
          f"= ~{n*4*2/1e6:.1f} MB")
    print(f"  💡 Fused saves ~50% HBM traffic")

    result = bench_compare(
        {
            "Triton fused bias+GELU": lambda: fused_bias_gelu(x, bias),
            "PyTorch (non-fused)": lambda: torch.nn.functional.gelu(
                x + bias, approximate="tanh"),
        },
        flops=n * 10,         # ~add + 3*mul + 2*add + mul + tanh
        bytes_accessed=n * 4 * (2 if True else 4),  # fused: 2 reads + 1 write
        dtype="fp32",
    )
    print_compare_report(result)


# PERFORMANCE NOTES
# =================
# - Fused bias+GELU: 在寄存器中完成 add + GELU, 只在首尾读写 HBM.
# - 对比非融合版本 (x+bias → temp, GELU(temp) → output): 节省 ~50% HBM 读写.
# - GELU 用 tanh 近似 (和 08_gelu.py 相同), 精度损失 < 0.01%.
# - 可以进一步融合: x+bias+GELU+dropout 在 transformer FFN 中也是一个常见融合点.
# - 寄存器压力: 本 kernel 用 ~30 个寄存器, 离 255 个上限还很远.

if __name__ == "__main__":
    main()
