"""
49_gelu_accurate.py — Exact GELU (erf-based) & Comparison with tanh Approximation

学习目标:
  - 掌握精确 GELU = x * Φ(x) = x * 0.5 * (1 + erf(x/sqrt(2))) 的 GPU 实现
  - 对比精确 GELU 和 tanh 近似 (08_gelu.py) 的精度/速度 trade-off
  - 理解特殊函数 (erf) 在 GPU 上的实现代价

GELU 两种定义:
  ┌──────────────────┬──────────────────────────────────────────────┐
  │ Exact GELU       │ x * 0.5 * (1 + erf(x / sqrt(2)))            │
  │ Tanh Approx      │ x * 0.5 * (1 + tanh(sqrt(2/π)*(x+0.044715*x³))) │
  ├──────────────────┼──────────────────────────────────────────────┤
  │ 最大误差          │ 0 (定义) vs ~0.0001 (对于 x∈[-3,3])          │
  │ GPU 延迟          │ ~100 cycles (erf 需要查表+插值)              │
  │                  │ ~30 cycles (tanh via MUFU)                   │
  │ 适用场景          │ 训练 (需要精确梯度)                          │
  │                  │ 推理 (tanh 更快, 精度损失可忽略)              │
  └──────────────────┴──────────────────────────────────────────────┘

为什么 BERT/GPT 用 tanh 近似:
  1. 推理速度要求: tanh 比 erf 快 3-4x
  2. 精度差异 < 0.1%, 对下游任务影响可忽略
  3. 训练时精确 GELU 的梯度差异也很小

erf 的定义:
  erf(z) = (2/√π) ∫₀ᶻ exp(-t²) dt
  - 没有闭合解, 需要数值近似 (多项式, 查表, 或牛顿法)
  - GPU MUFU 单元通常支持 tanh/sigmoid/exp, 但不支持 erf

运行: python phase1_fundamentals/49_gelu_accurate.py
"""

import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch, triton, triton.language as tl
from utils.profiler import bench_compare, print_compare_report


@triton.jit
def gelu_exact_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2))).
    Triton 中没有内置 erf, 用多项式近似实现.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # [GPU] Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    # tl.math.erf is available in Triton 3.6+
    z = x * 0.7071067811865475  # x / sqrt(2)
    erf_z = tl.math.erf(z)
    result = x * 0.5 * (1.0 + erf_z)

    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def gelu_tanh_approx_kernel(x_ptr, output_ptr, n_elements,
                              BLOCK_SIZE: tl.constexpr):
    """
    Tanh-approximate GELU (更快的版本).
    gelu(x) ≈ x * 0.5 * (1 + tanh(sqrt(2/π)*(x + 0.044715*x³)))
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # [GPU] GELU ≈ x * sigmoid(2*inner) via MUFU (~16 cycles)
    # 推导: tanh(z)=2*sigmoid(2z)-1, GELU=x*0.5*(1+tanh(inner))=x*sigmoid(2*inner)
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    result = x * tl.sigmoid(2.0 * inner)

    tl.store(output_ptr + offsets, result, mask=mask)


def gelu_exact(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    gelu_exact_kernel[grid](x, output, n, BLOCK_SIZE=1024)
    return output


def gelu_tanh_approx(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    gelu_tanh_approx_kernel[grid](x, output, n, BLOCK_SIZE=1024)
    return output


def main():
    print("=" * 60)
    print("49_gelu_accurate — Exact GELU vs Tanh Approximation")
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.manual_seed(42)

    # Correctness of exact GELU
    for name, size in [("small ", 256), ("medium", 65536), ("large ", 1048576)]:
        x = torch.randn(size, device="cuda")
        y_t = gelu_exact(x)
        y_r = torch.nn.functional.gelu(x, approximate="none")
        max_diff = (y_t - y_r).abs().max().item()
        print(f"  [exact {name}] size={size:8d}  max_diff={max_diff:.2e}  "
              f"{'✅' if max_diff < 1e-5 else '❌'}")

    # Compare exact vs tanh approximation
    print("\n--- Exact GELU vs Tanh Approximation ---")
    x = torch.linspace(-3, 3, 13, device="cuda")
    y_exact = gelu_exact(x)
    y_tanh = gelu_tanh_approx(x)
    y_ref = torch.nn.functional.gelu(x, approximate="none")
    print(f"  {'x':>8s}  {'Exact':>10s}  {'Tanh':>10s}  {'|err|':>10s}")
    for i in range(len(x)):
        err = abs(y_tanh[i].item() - y_exact[i].item())
        print(f"  {x[i].item():8.3f}  {y_exact[i].item():10.6f}  "
              f"{y_tanh[i].item():10.6f}  {err:10.2e}")

    print(f"\n  Max |exact - tanh| error: "
          f"{(y_exact - y_tanh).abs().max().item():.2e}")

    # Performance: exact vs tanh
    print("\n--- Performance: Exact vs Tanh ---")
    x = torch.randn(16777216, device="cuda", dtype=torch.float32)
    import time
    for _ in range(10): gelu_exact(x); gelu_tanh_approx(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100): gelu_exact(x)
    torch.cuda.synchronize()
    t_exact = (time.perf_counter() - t0) / 100 * 1000
    t0 = time.perf_counter()
    for _ in range(100): gelu_tanh_approx(x)
    torch.cuda.synchronize()
    t_tanh = (time.perf_counter() - t0) / 100 * 1000
    t0 = time.perf_counter()
    for _ in range(100): torch.nn.functional.gelu(x, approximate="tanh")
    torch.cuda.synchronize()
    t_pt = (time.perf_counter() - t0) / 100 * 1000
    print(f"  Exact GELU (ours):    {t_exact:.4f} ms")
    print(f"  Tanh approx (ours):   {t_tanh:.4f} ms")
    print(f"  PyTorch gelu(tanh):   {t_pt:.4f} ms")
    print(f"  Speedup (tanh/exact): {t_exact/t_tanh:.2f}x")


# PERFORMANCE NOTES
# =================
# - Exact GELU 需要 erf 多项式近似 (5 次多项式 + exp), ~50-100 cycles.
# - Tanh GELU 用 MUFU 硬件的 tanh 指令, ~16-30 cycles.
# - 训练用 Exact GELU (精度), 推理用 Tanh GELU (速度).
# - 和 08_gelu.py (tanh 近似) 的关系: 08 实现了 tanh 近似,
#   本文件补充精确版本和两种方案的对比分析.

if __name__ == "__main__":
    main()
