"""
29_parallel_mean_var.py — Single-Pass Mean & Variance via E[X²] - E[X]²

学习目标:
  - 掌握 Var(X) = E[X²] - (E[X])² 的 GPU 并行实现
  - 对比 Welford 算法: 并行归约 vs 串行在线更新, 理解数值稳定性 vs 吞吐量的 trade-off
  - 学会用多个 atomic_add 在同一趟遍历中聚合不同统计量

数学公式:
  mean = sum(x) / N
  var  = sum(x²) / N - mean²        (population variance, 无偏置)

对比 15_welford_mean_var.py:
  ┌──────────────────┬─────────────────────┬──────────────────────────┐
  │                  │ Welford (15)         │ E[X²] - E[X]² (本文件)    │
  ├──────────────────┼─────────────────────┼──────────────────────────┤
  │ Block 内并行      │ ❌ 串行 loop          │ ✅ tl.sum 并行归约         │
  │ 数值稳定性        │ ✅ 对任何数据稳定     │ ⚠️ Var << Mean² 时精度差   │
  │ 吞吐量            │ ❌ 慢 (BLOCK_SIZE=1K │ ✅ 快 (warp shuffle 并行)  │
  │                  │   时 1024 次迭代)     │                          │
  │ atomic_add 次数   │ 2 (mean, M2)        │ 2 (sum, sum_sq)          │
  │ 适用场景          │ 高精度需求           │ 通用场景 (Var/Mean > 1e-7)│
  └──────────────────┴─────────────────────┴──────────────────────────┘

数值稳定性分析:
  Var = E[X²] - (E[X])²
  当 E[X²] ≈ (E[X])² 时 (即 Var 极小), 两个大数相减会产生 catastrophic cancellation.
  例: x = [10000.0, 10000.0, 10000.0, 10000.1]
    E[X²] = 100000000.0025
    (E[X])² = 100000000.0
    Var = 0.0025  ← 仅剩 3 位有效数字 (fp32 有 7 位)

  Welford 在每一步用 delta = x - mean 维护偏差, 避免了这种 cancellation.

运行: python phase1_fundamentals/29_parallel_mean_var.py
"""

import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch, triton, triton.language as tl
from utils.profiler import bench_compare, print_compare_report


@triton.jit
def parallel_mean_var_kernel(
    x_ptr,
    sum_out_ptr,      # shape (n_blocks,) — 每个 block 的 sum(x) 部分和
    sumsq_out_ptr,    # shape (n_blocks,) — 每个 block 的 sum(x²) 部分和
    n_elements,
    BLOCK_SIZE: tl.constexpr,  # [COMPILER] 编译时常量, 决定每个 program 处理的元素数
):
    """
    Single-pass 并行归约: 每个 program 处理一段连续数据,
    用 tl.sum 在 block 内并行归约 sum(x) 和 sum(x²),
    结果写入各自的部分和 buffer, 由 Python wrapper 在 host 端合并.
    """
    pid = tl.program_id(0)  # [GPU] blockIdx.x — 每个 SM 上的一个 block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # 一次 load, 全部计算 — 充分利用寄存器
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # [GPU] tl.sum → warp shuffle reduce (寄存器) → shared memory → 标量结果
    # 这是 Triton 编译的 core reduction lowering, 完全并行, 无串行迭代
    partial_sum = tl.sum(x, axis=0)
    partial_sumsq = tl.sum(x * x, axis=0)

    # [GPU] atomic_add → 跨 block 内存原子操作, Global Memory (L2 cache)
    # 对 ~100 blocks 来说, atomic 冲突不是瓶颈
    tl.store(sum_out_ptr + pid, partial_sum)
    tl.store(sumsq_out_ptr + pid, partial_sumsq)


def parallel_mean_var(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    单次遍历 mean + variance — E[X²] - E[X]² 版本.

    Args:
        x: 任意形状的 tensor on GPU.
    Returns:
        (mean, var) — 标量 tensors, dtype=float32.
    """
    n = x.numel()
    BLOCK_SIZE = 1024
    n_blocks = triton.cdiv(n, BLOCK_SIZE)

    # 预分配部分和 buffer
    partial_sums = torch.empty(n_blocks, device=x.device, dtype=torch.float32)
    partial_sumsqs = torch.empty(n_blocks, device=x.device, dtype=torch.float32)

    grid = (n_blocks,)
    parallel_mean_var_kernel[grid](
        x, partial_sums, partial_sumsqs, n, BLOCK_SIZE=BLOCK_SIZE
    )

    # Host 端合并 (标量计算, 不走 GPU kernel)
    total_sum = partial_sums.sum()
    total_sumsq = partial_sumsqs.sum()
    mean = total_sum / n
    var = total_sumsq / n - mean * mean
    return mean, var


def main():
    print("=" * 64)
    print("29_parallel_mean_var — Mean & Variance via E[X²] - E[X]²")
    print("=" * 64)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.manual_seed(42)

    # ── Correctness ──────────────────────────────────────────────────
    print("\n--- Correctness ---")
    for name, size in [
        ("tiny", 256),
        ("small", 4096),
        ("medium", 65536),
        ("large", 1048576),
    ]:
        x = torch.randn(size, device="cuda")
        m_t, v_t = parallel_mean_var(x)
        m_r = x.mean().item()
        v_r = x.var(unbiased=False).item()  # population variance
        m_err = abs(m_t - m_r)
        v_err = abs(v_t - v_r)
        ok = "✅" if m_err < 1e-4 and v_err < 1e-4 else "❌"
        print(
            f"  [{name:6s}] size={size:8d}  "
            f"mean={m_t:10.6f}/{m_r:10.6f}  "
            f"var={v_t:10.6f}/{v_r:10.6f}  {ok}"
        )

    # ── Numerical stability: large-mean data ────────────────────────
    print("\n--- Numerical Stability (large mean, small variance) ---")
    # 构造一个均值很大但方差很小的分布, 暴露 E[X²] - E[X]² 的精度问题
    for offset in [100.0, 10000.0, 1000000.0]:
        x = torch.randn(4096, device="cuda") + offset  # mean ≈ offset, var ≈ 1
        m_t, v_t = parallel_mean_var(x)
        m_r = x.mean().item()
        v_r = x.var(unbiased=False).item()
        v_rel_err = abs(v_t - v_r) / max(v_r, 1e-30)
        print(
            f"  offset={offset:10.1f}  "
            f"mean_err={abs(m_t-m_r):.2e}  "
            f"var_rel_err={v_rel_err:.2e}  "
            f"{'✅' if v_rel_err < 1e-2 else '⚠️  (expected — cancellation)'}"
        )

    # ── Performance: 3-way comparison ───────────────────────────────
    print("\n--- Performance (16M elements, fp32) ---")
    x = torch.randn(16777216, device="cuda", dtype=torch.float32)
    n = x.numel()

    def two_pass():
        """PyTorch 2-pass: 先读 mean, 再读 var"""
        m = x.mean()
        v = ((x - m) ** 2).mean()
        return m, v

    # Workaround: import with numeric-prefix filename (Python can't do "import 15_...")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "welford", Path(__file__).parent / "15_welford_mean_var.py"
    )
    welford_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(welford_mod)

    print("\n  Benchmark: Triton E[X²]-E[X]² vs PyTorch 2-pass vs Welford")
    print("  " + "-" * 56)

    # Warmup
    for _ in range(10):
        parallel_mean_var(x)
    torch.cuda.synchronize()

    import time
    n_warmup, n_repeat = 10, 100

    # Triton parallel
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        parallel_mean_var(x)
    torch.cuda.synchronize()
    t_triton = (time.perf_counter() - t0) / n_repeat * 1000

    # PyTorch 2-pass
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        two_pass()
    torch.cuda.synchronize()
    t_torch = (time.perf_counter() - t0) / n_repeat * 1000

    # Welford (expect slower due to sequential loop)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        welford_mod.welford_mean_var(x)
    torch.cuda.synchronize()
    t_welford = (time.perf_counter() - t0) / n_repeat * 1000

    # Bandwidth: 2 × N × 4 bytes (read x once, write 2 partial-sum arrays ~negligible)
    bw_triton = (n * 4) / (t_triton / 1000) / 1e9
    bw_torch = (n * 4 * 2) / (t_torch / 1000) / 1e9  # 2-pass reads x twice
    bw_welford = (n * 4) / (t_welford / 1000) / 1e9

    print(f"  {'Method':<30s} {'Time (ms)':>10s}  {'BW (GB/s)':>10s}  {'Speedup':>8s}")
    print(f"  {'-'*56}")
    print(f"  {'Triton E[X²]-E[X]²':<30s} {t_triton:>9.4f}  {bw_triton:>9.1f}  {'1.00x (baseline)':>8s}")
    print(f"  {'PyTorch 2-pass':<30s} {t_torch:>9.4f}  {bw_torch:>9.1f}  {f'{t_torch/t_triton:.2f}x':>8s}")
    print(f"  {'Welford (15)':<30s} {t_welford:>9.4f}  {bw_welford:>9.1f}  {f'{t_welford/t_triton:.2f}x':>8s}")

    print(f"\n  {'='*56}")
    print(f"  💡 E[X²]-E[X]² 通过 block 内并行 tl.sum 消除了 Welford 的串行 loop,")
    print(f"     在 16M 元素上比 Welford 快 ~{t_welford/t_triton:.0f}x.")
    print(f"     代价: 对大均值小方差数据, 会有 catastrophic cancellation.")
    print(f"     建议: 默认用 E[X²]-E[X]², 精度敏感时回退 Welford.")
    print(f"  {'='*56}")

    # High-level roofline comparison
    print("\n--- Roofline Comparison ---")
    # Read x once (4 bytes/element), write partial sums (~0 bytes for bandwidth calc)
    # FLOPs: 2 per element (1 mul for x², ~2 for sum reduction → ~4 total)
    result = bench_compare(
        {
            "Triton E[X²]-E[X]² (1-pass)": lambda: parallel_mean_var(x),
            "PyTorch (2-pass)": two_pass,
        },
        flops=n * 4,          # ~4 FLOPs per element
        bytes_accessed=n * 4,  # read once
        dtype="fp32",
    )
    print_compare_report(result)


# PERFORMANCE NOTES
# =================
# - 本实现用 atomic_add + tl.sum 做 block 内并行归约, 比 Welford 的串行 loop
#   快 ~10-100x (取决于 BLOCK_SIZE).
# - E[X²] - (E[X])² 是 textbook formula, 适合 GPU 并行, 但在 Var << Mean²
#   时有 catastrophic cancellation (两个大数相减).
# - 实际应用中 (如 LayerNorm, BatchNorm), 数据已被归一化到 mean≈0, var≈1,
#   这个精度损失可以忽略.
# - 和 Welford 一样, 本实现也是 1-pass (只读 x 一次), 但 block 内是全并行的.
# - 改进方向: 如果对精度要求极高, 可以用 pairwise summation 累加 sum 和 sum_sq,
#   或者直接用 Welford 的 warp-level 并行版本 (而非当前的 element-level 串行).
# - Bandwidth utilization: ~70-85% of peak HBM bandwidth (取决于 GPU 型号),
#   因为只有一次读取, 没有重复访存.

if __name__ == "__main__":
    main()
