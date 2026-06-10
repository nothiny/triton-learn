"""
11_geglu.py — Fused GeGLU (GELU Gated Linear Unit) kernel

学习目标：
  - 对比 GeGLU vs SwiGLU 的实现差异 (GELU vs SiLU)
  - 理解 gated activation 的设计空间
  - 掌握如何复用已有 GELU 实现构建新 kernel

数学公式:
  GeGLU(gate, up) = gate * GELU(up)

  其中 GELU(x) ≈ 0.5*x*(1+tanh(√(2/π)*(x+0.044715*x³)))

和 SwiGLU 的对比:
  - SwiGLU: gate * SiLU(up)  — Llama, Mistral, Qwen 使用
  - GeGLU:  gate * GELU(up)  — 部分 BERT 变体使用
  - SiLU 更简单 (sigmoid), GELU 更复杂 (tanh approx)
  - 实践中 SwiGLU 更流行 (更快, 效果相当)

Liger 对比:
  - liger_geglu: 生产级 Triton 实现
  - 参数顺序: liger_geglu(up, gate) — 参数相反!

运行: python phase1_fundamentals/11_geglu.py
"""

import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.profiler import bench_compare, print_compare_report
from benchmarks.references.liger_ref import get_liger_geglu


@triton.jit
def geglu_kernel(
    gate_ptr, up_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused GeGLU: output[i] = gate[i] * GELU(up[i])

    单 kernel 完成 GELU 激活 + gate 乘法.
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0)

    # ---- GELU(up) = 0.5 * up * (1 + tanh(√(2/π) * (up + 0.044715 * up³))) ----
    up3 = up * up * up
    inner = 0.7978845608028654 * (up + 0.044715 * up3)
    tanh_val = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
    gelu_up = 0.5 * up * (1.0 + tanh_val)

    # GeGLU = gate * GELU(up)
    output = gate * gelu_up
    tl.store(output_ptr + offsets, output, mask=mask)


def geglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused GeGLU: gate * GELU(up)."""
    assert gate.shape == up.shape
    output = torch.empty_like(gate)
    n = gate.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    geglu_kernel[grid](gate, up, output, n, BLOCK_SIZE=1024)
    return output


def main():
    print("=" * 60)
    print("11_geglu — Fused GeGLU Activation")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    torch.manual_seed(42)

    for name, shape in [("small", (256, 512)), ("medium", (2048, 4096))]:
        gate = torch.randn(*shape, device="cuda")
        up = torch.randn(*shape, device="cuda")

        out_triton = geglu(gate, up)
        out_torch = gate * torch.nn.functional.gelu(up, approximate="tanh")

        max_diff = (out_triton - out_torch).abs().max().item()
        print(f"  [{name}] shape={shape}  max_diff={max_diff:.2e}  "
              f"{'✅' if max_diff < 1e-4 else '❌'}")

    print("\n--- Performance ---")
    gate = torch.randn(2048, 8192, device="cuda", dtype=torch.float32)
    up = torch.randn(2048, 8192, device="cuda", dtype=torch.float32)

    implementations = {
        "Triton Fused (ours)": lambda: geglu(gate, up),
        "PyTorch Unfused": lambda: gate * torch.nn.functional.gelu(up, approximate="tanh"),
    }
    liger_geglu_fn = get_liger_geglu()
    if liger_geglu_fn:
        implementations["Liger Fused (SotA)"] = lambda: liger_geglu_fn(gate, up)

    n = gate.numel()
    result = bench_compare(implementations, flops=n * 10, bytes_accessed=n * 3 * 4,
                           dtype="fp32")
    print_compare_report(result)


# PERFORMANCE NOTES
# =================
# - GeGLU ≈ 10 FLOPs/elt vs SwiGLU ≈ 6 FLOPs/elt (GELU 比 SiLU 多 ~4 FLOPs)
# - 但两者都是 memory-bound → 实际运行时间几乎相同
# - GeGLU 在生产中不如 SwiGLU 流行:
#   - GELU 计算量大 40%, 但效果没显著提升
#   - SiLU 的 sigmoid 硬件实现更高效
# - 建议: 新项目优先用 SwiGLU

if __name__ == "__main__":
    main()
