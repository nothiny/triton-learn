"""
10_swiglu.py — Fused SwiGLU (Sigmoid Linear Unit with Gating)

学习目标：
  - 理解 SwiGLU 在现代 LLM 中的角色（Llama, Mistral, Qwen 都用它）
  - 掌握 fused gated activation 的实现
  - 对比分离版 vs 融合版的性能差异

数学公式:
  SwiGLU(gate, up) = gate * SiLU(up)
  其中 SiLU(up) = up * sigmoid(up)

  所以完整公式: SwiGLU(gate, up) = gate * up * sigmoid(up)

在 LLM 中的使用:
  - 标准 FFN:  output = Linear↓(ReLU(Linear↑(x)))
  - SwiGLU FFN: output = Linear↓(SiLU(Linear_gate(x)) * Linear_up(x))
  - SwiGLU 比 ReLU 效果更好: 平滑 + gating 机制提供更强的表达能力

Fusion 收益:
  不 fusion: SiLU(up) → write HBM → read → gate * ...
  Fusion:    gate * up * sigmoid(up) 全程在寄存器中

对比:
  - Liger Kernel (linkedin): 生产级 Triton SwiGLU
  - PyTorch: gate * F.silu(up)

运行: python phase1_fundamentals/10_swiglu.py
"""

import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.profiler import bench_compare, print_compare_report
from benchmarks.references.liger_ref import get_liger_swiglu


@triton.jit
def swiglu_kernel(
    gate_ptr,    # gate 输入
    up_ptr,      # up 输入 (经过 SiLU 激活)
    output_ptr,  # 输出: gate * SiLU(up)
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused SwiGLU: output[i] = gate[i] * up[i] * sigmoid(up[i])

    单 kernel 完成 SiLU 激活 + gate 乘法。
    如果不 fusion: 需要 SiLU(up) → 写 HBM → 读回 → gate * ...
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # 加载 gate 和 up
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0)

    # SwiGLU: gate * up * sigmoid(up)
    # [COMPILER] tl.sigmoid 利用 MUFU 硬件, 三个操作 (load gate, load up, fused compute)
    # 全部在寄存器中完成, 无需 HBM 中间存储
    output = gate * up * tl.sigmoid(up)

    tl.store(output_ptr + offsets, output, mask=mask)


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    Fused SwiGLU 激活: output = gate * SiLU(up).

    Args:
        gate: gate 张量
        up: up 张量 (会经过 SiLU 激活)

    Returns:
        gate * up * sigmoid(up)

    NOTE: 标准 SwiGLU 约定是 SwiGLU(gate, up) = gate * SiLU(up).
          Liger 的约定是 swiglu(up, gate) — 参数顺序相反!
    """
    assert gate.shape == up.shape, f"Shape mismatch: {gate.shape} vs {up.shape}"
    output = torch.empty_like(gate)
    n_elements = gate.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    swiglu_kernel[grid](gate, up, output, n_elements, BLOCK_SIZE=1024)
    return output


def main():
    print("=" * 60)
    print("10_swiglu — Fused SwiGLU Activation")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # ---- 正确性测试 ----
    torch.manual_seed(42)

    test_cases = [
        ("normal", 2048, 4096),
        ("large", 4096, 8192),
        ("small", 256, 512),
    ]

    for name, rows, cols in test_cases:
        gate = torch.randn(rows, cols, device="cuda")
        up = torch.randn(rows, cols, device="cuda")

        out_triton = swiglu(gate, up)
        # PyTorch unfused reference: gate * SiLU(up)
        out_torch = gate * torch.nn.functional.silu(up)

        max_diff = (out_triton - out_torch).abs().max().item()
        status = "✅" if max_diff < 1e-4 else "❌"
        print(f"  [{name}] shape=({rows}, {cols})  max_diff={max_diff:.2e}  {status}")

    # ---- 性能对比: Triton vs PyTorch vs Liger ----
    print("\n--- Performance ---")

    gate = torch.randn(2048, 8192, device="cuda", dtype=torch.float32)
    up = torch.randn(2048, 8192, device="cuda", dtype=torch.float32)

    # Unfused baseline: 2 ops
    def unfused(g, u):
        return g * torch.nn.functional.silu(u)

    implementations = {
        "Triton Fused (ours)": lambda: swiglu(gate, up),
        "PyTorch Unfused": lambda: unfused(gate, up),
    }

    # Liger SwiGLU (note: parameter order is swapped!)
    liger_swiglu_fn = get_liger_swiglu()
    if liger_swiglu_fn:
        # liger convention: swiglu(up, gate) — our wrapper already handles this
        implementations["Liger Fused (SotA)"] = lambda: liger_swiglu_fn(gate, up)

    n_elements = gate.numel()
    # sigmoid(~4) + 2*mul = ~6 FLOPs per element
    flops_total = n_elements * 6
    # gate(4B) + up(4B) + out(4B) = 12 bytes per element
    bytes_total = n_elements * 3 * 4

    result = bench_compare(
        implementations,
        flops=flops_total,
        bytes_accessed=bytes_total,
        dtype="fp32",
    )
    print_compare_report(result)


# PERFORMANCE NOTES
# =================
# - SwiGLU 是 memory-bound: ~6 FLOPs / 12 bytes = 0.5 FLOP/byte
# - Fusion 的核心收益: 避免 SiLU(up) 中间结果写回 HBM
#   - 不 fusion: SiLU(up) → write (4B/elt) → read (4B/elt) → gate * ...
#     多了 8 bytes/elt 的 HBM 流量 (= 40% 额外带宽)
# - SwiGLU vs ReLU:
#   - ReLU FFN: output = W2(ReLU(W1(x)))  — 单一激活
#   - SwiGLU FFN: output = W2(SiLU(W_gate(x)) * W_up(x))  — gated 激活
#   - SwiGLU 多了一个矩阵乘和 gating, 但效果显著提升
# - LLM 中的典型维度 (Llama-7B):
#   - hidden_dim = 4096, intermediate_dim = 11008
#   - gate/up 都是 (batch, seq_len, 11008)
# - [COMPILER] 三个操作 (load gate, load up, compute) 在 Triton 编译后
#   会融合为单个 coalesced 内存访问 + 寄存器计算序列

if __name__ == "__main__":
    main()
