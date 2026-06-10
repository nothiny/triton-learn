"""
09_residual_add_norm.py — Fused Residual Add + LayerNorm kernel

学习目标：
  - 理解 Transformer 中最常见的 operator fusion: residual + norm
  - 体会 fusion 如何减少 HBM round-trip
  - 掌握多输入 kernel 的设计模式

背景 — Transformer 中的残差连接:
  每个 Transformer block 的输出都是:
    output = LayerNorm(x + residual)

  其中:
    - x: 当前 sublayer 的输出 (如 FFN 或 Attention)
    - residual: 残差连接的输入 (skip connection)

  [COMPILER] 如果不 fusion, 这需要 2 个 kernel:
    1. temp = x + residual         → write temp to HBM
    2. LayerNorm(temp, w, b)       → read temp from HBM (×2-3 passes)

  Fused kernel 避免写 temp 到 HBM:
    1. 在寄存器中计算 x + residual → 直接传递给 norm 逻辑 → 只写 output

Fusion 收益分析:
  假设 fp32, N 个元素:
    不 fusion:  (read 2N) + (write N) + (read 2-3N) + (write N) = 6-7N × 4B
    Fusion:     (read 3N) + (write N)                               = 4N × 4B
    减少约 25-40% 的 HBM 流量

运行: python phase1_fundamentals/20_residual_add_norm.py
"""

import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).parent.parent))
import importlib

from utils.profiler import bench_compare, print_compare_report
from benchmarks.references.liger_ref import get_liger_ln

# Load our LayerNorm for unfused comparison
def _load_fn(path, name):
    spec = importlib.util.spec_from_file_location(
        path.replace("/", "_").replace(".", "_"), path + ".py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, name)

layer_norm_fn = _load_fn("phase1_fundamentals/16_layer_norm", "layer_norm")


@triton.jit
def residual_add_norm_kernel(
    x_ptr,         # 当前 sublayer 输出: (N_ROWS, N_COLS)
    residual_ptr,  # 残差输入: (N_ROWS, N_COLS) — 与 x 同 shape
    weight_ptr,    # γ (scale), shape: (N_COLS,)
    bias_ptr,      # β (shift), shape: (N_COLS,)
    output_ptr,    # 输出: (N_ROWS, N_COLS)
    n_cols,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused: output[row, :] = LayerNorm(x[row,:] + residual[row,:])

    每个 program 处理一行。
    3-pass over columns, 在寄存器中计算 x+residual。
    """
    row_idx = tl.program_id(axis=0)
    row_start = row_idx * n_cols
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # ---- Pass 1: 计算 mean(x + residual) ----
    accum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = row_start + block_start + col_offsets
        mask = (block_start + col_offsets) < n_cols
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        # FUSION: x + residual 在寄存器中完成, 不写回 HBM
        accum += (x + r)
    global_mean = tl.sum(accum, axis=0) / n_cols

    # ---- Pass 2: 计算 variance(x + residual) ----
    sq_accum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = row_start + block_start + col_offsets
        mask = (block_start + col_offsets) < n_cols
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        diff = (x + r) - global_mean
        sq_accum += diff * diff
    global_var = tl.sum(sq_accum, axis=0) / n_cols

    # ---- Pass 3: 归一化 + affine + 写回 ----
    inv_std = 1.0 / tl.sqrt(global_var + eps)
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = row_start + block_start + col_offsets
        mask = (block_start + col_offsets) < n_cols
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + block_start + col_offsets, mask=mask, other=0.0)
        b = tl.load(bias_ptr + block_start + col_offsets, mask=mask, other=0.0)

        # LayerNorm: γ * (x+r - μ) / σ + β
        normalized = w * ((x + r) - global_mean) * inv_std + b
        tl.store(output_ptr + offsets, normalized, mask=mask)


def residual_add_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Fused residual add + LayerNorm.

    Args:
        x: 当前 sublayer 输出 (N_ROWS, N_COLS)
        residual: 残差输入 (N_ROWS, N_COLS)
        weight: scale γ (N_COLS,)
        bias: shift β (N_COLS,)
        eps: 数值稳定常数

    Returns:
        LayerNorm(x + residual), shape (N_ROWS, N_COLS)
    """
    assert x.shape == residual.shape, f"Shape mismatch: {x.shape} vs {residual.shape}"
    output = torch.empty_like(x)
    n_rows, n_cols = x.shape
    grid = (n_rows,)
    residual_add_norm_kernel[grid](
        x, residual, weight, bias, output, n_cols, eps=eps, BLOCK_SIZE=1024
    )
    return output


def main():
    print("=" * 60)
    print("09_residual_add_norm — Fused Residual + LayerNorm")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # ---- 正确性测试 ----
    N_ROWS, N_COLS = 2048, 4096
    torch.manual_seed(42)
    x = torch.randn(N_ROWS, N_COLS, device="cuda")
    residual = torch.randn(N_ROWS, N_COLS, device="cuda")
    weight = torch.randn(N_COLS, device="cuda")
    bias = torch.randn(N_COLS, device="cuda")
    eps = 1e-5

    # Fused: LayerNorm(x + residual)
    out_triton = residual_add_norm(x, residual, weight, bias, eps)

    # Unfused reference: split into two ops
    tmp = x + residual
    out_torch = torch.nn.functional.layer_norm(tmp, [N_COLS], weight, bias, eps)

    max_diff = (out_triton - out_torch).abs().max().item()
    print(f"  Shape: ({N_ROWS}, {N_COLS})")
    print(f"  Max diff: {max_diff:.6e}")
    print(f"  Status: {'✅ PASS' if max_diff < 1e-3 else '❌ FAIL'}")

    # ---- 性能对比: Fused vs Unfused ----
    print("\n--- Performance ---")

    def unfused_ref():
        """PyTorch 'standard' approach: two separate ops."""
        tmp = x + residual
        return torch.nn.functional.layer_norm(tmp, [N_COLS], weight, bias, eps)

    implementations = {
        "Triton Fused (ours)": lambda: residual_add_norm(x, residual, weight, bias, eps),
    }

    # Unfused PyTorch (add + layernorm = 2 kernel calls)
    implementations["PyTorch Unfused"] = unfused_ref

    # Unfused Triton (add + our layernorm = 2 kernel calls)
    if layer_norm_fn is not None:
        def triton_unfused():
            tmp = x + residual
            return layer_norm_fn(tmp, weight, bias, eps)
        implementations["Triton Unfused"] = triton_unfused

    # Liger LayerNorm (unfused: add + liger LN)
    liger_ln = get_liger_ln()
    if liger_ln:
        def liger_unfused():
            tmp = x + residual
            return liger_ln(tmp, weight, bias, eps)
        implementations["Liger Unfused"] = liger_unfused

    n_elements = x.numel()
    # mean + var + norm + affine ≈ 8 FLOPs per element (same as LayerNorm)
    flops_total = n_elements * 9  # +1 for the add
    # Fused: x(4) + r(4) + w(4) + b(4) read (w/b once per row effectively) + out(4) write
    bytes_total = n_elements * 5 * 4  # 5 streams: x, r, w, b, out

    result = bench_compare(
        implementations,
        flops=flops_total,
        bytes_accessed=bytes_total,
        dtype="fp32",
    )
    print_compare_report(result)


# PERFORMANCE NOTES
# =================
# - Residual + LayerNorm 是 Transformer 中最常见的 pattern
#   - 每个 decoder block 有 2 次: 一次在 attention 后, 一次在 FFN 后
#   - Llama-7B: 32 blocks × 2 = 64 次 residual+norm per forward pass
# - Fusion 的好处:
#   1. 减少 HBM round-trip: 不需要写临时 x+residual
#   2. 减少 kernel launch overhead: 1 kernel vs 2
#   3. 保留 fp32 精度: x+residual 在寄存器中是 fp32, 写 HBM 会用 fp16
# - HBM 流量对比 (fp32, N elements):
#   - Unfused:  tmp = x+r (3N read/write + N write) = 4N
#               LN(tmp) (3N read + N write)          = 4N
#               Total                                 = 8N × 4B = 32N bytes
#   - Fused:    LN(x+r) inline (5N read + N write)   = 6N × 4B = 24N bytes
#   减少 25% HBM 流量!
# - 实际加速比通常 1.1-1.3x (因为 LayerNorm 本身 memory-bound, fusion
#   主要节省的是 add 的 HBM 带宽)
# - 进一步优化: 可以用 Welford (1-pass) 或 shared memory 减少 x+r 的多次读取
# - [COMPILER] (x+r) 在寄存器中不产生中间 store → LLVM IR 层面被
#   mem2reg / GVN pass 优化掉

if __name__ == "__main__":
    main()
