"""
GPU kernel performance profiling utilities.

Provides precise GPU timing using CUDA events, TFLOPS / bandwidth
calculation, and bottleneck analysis for roofline model reasoning.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class ProfileResult:
    """Result of a single kernel profiling run."""

    name: str
    time_ms: float
    time_std_ms: float = 0.0
    tflops: float = 0.0
    bandwidth_gb_s: float = 0.0
    flops_count: int = 0
    bytes_count: int = 0
    bottleneck: str = ""
    occupancy_pct: float = 0.0

    def __repr__(self) -> str:
        lines = [f"  {self.name}"]
        lines.append(f"    Time:      {self.time_ms:.4f} ms ± {self.time_std_ms:.4f}")
        if self.tflops > 0:
            lines.append(f"    TFLOPS:    {self.tflops:.4f}")
        if self.bandwidth_gb_s > 0:
            lines.append(f"    Bandwidth: {self.bandwidth_gb_s:.2f} GB/s")
        if self.bottleneck:
            lines.append(f"    Bottleneck: {self.bottleneck}")
        return "\n".join(lines)


@dataclass
class GPUInfo:
    """Static GPU capability information for roofline analysis."""

    name: str
    peak_fp16_tflops: float  # Theoretical peak TFLOPS (fp16 tensor core)
    peak_fp32_tflops: float  # Theoretical peak TFLOPS (fp32)
    hbm_bandwidth_gb_s: float  # HBM bandwidth in GB/s
    sm_count: int
    max_shared_mem_per_sm: int  # bytes
    max_registers_per_sm: int

    # Pre-defined values for common GPUs
    _KNOWN: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._KNOWN = {}

    @classmethod
    def detect(cls) -> "GPUInfo":
        """Auto-detect GPU and return its specs."""
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA-capable GPU detected")

        name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)

        # Known GPU specs (approximate peak values)
        known = {
            "H100": {
                "peak_fp16_tflops": 989.0,  # with sparsity: 1979
                "peak_fp32_tflops": 67.0,
                "hbm_bandwidth_gb_s": 3350.0,  # HBM3
            },
            "H800": {
                "peak_fp16_tflops": 989.0,
                "peak_fp32_tflops": 67.0,
                "hbm_bandwidth_gb_s": 3350.0,
            },
            "A100": {
                "peak_fp16_tflops": 312.0,  # with sparsity: 624
                "peak_fp32_tflops": 19.5,
                "hbm_bandwidth_gb_s": 2039.0,  # 80GB HBM2e
            },
            "A10": {
                "peak_fp16_tflops": 125.0,
                "peak_fp32_tflops": 31.2,
                "hbm_bandwidth_gb_s": 600.0,
            },
            "RTX 4090": {
                "peak_fp16_tflops": 330.0,
                "peak_fp32_tflops": 82.6,
                "hbm_bandwidth_gb_s": 1008.0,
            },
        }

        # Find matching GPU spec
        matched = None
        for key, specs in known.items():
            if key in name:
                matched = specs
                break

        if matched is None:
            # Fallback: derive from SM count and clock (rough estimate)
            print(f"  [WARN] Unknown GPU '{name}', using rough estimates")
            matched = {
                "peak_fp16_tflops": props.multi_processor_count * 15.0,
                "peak_fp32_tflops": props.multi_processor_count * 5.0,
                "hbm_bandwidth_gb_s": 900.0,  # conservative guess
            }

        return cls(
            name=name,
            peak_fp16_tflops=matched["peak_fp16_tflops"],
            peak_fp32_tflops=matched["peak_fp32_tflops"],
            hbm_bandwidth_gb_s=matched["hbm_bandwidth_gb_s"],
            sm_count=props.multi_processor_count,
            max_shared_mem_per_sm=getattr(props, "shared_memory_per_block_optin", 0),
            max_registers_per_sm=65536,  # common across Ampere/Hopper
        )


class KernelProfiler:
    """
    GPU kernel performance measurement tool.

    Wraps torch.cuda.Event for precise timing and computes derived
    metrics (TFLOPS, bandwidth utilization, roofline bottleneck).

    Usage::

        profiler = KernelProfiler(warmup=5, rep=100)
        results = profiler.bench(
            lambda: my_kernel(a, b),
            name="my_kernel",
            flops=2 * M * N * K,  # for GEMM
            bytes=(M*K + K*N + M*N) * element_size,
        )
        profiler.print_report(results)
    """

    def __init__(self, warmup: int = 5, rep: int = 100) -> None:
        """
        Args:
            warmup: Number of warmup iterations (not measured).
            rep: Number of measurement iterations.
        """
        self.warmup = warmup
        self.rep = rep
        self._gpu_info: Optional[GPUInfo] = None

    @property
    def gpu_info(self) -> GPUInfo:
        """Lazy-load GPU info on first access."""
        if self._gpu_info is None:
            self._gpu_info = GPUInfo.detect()
        return self._gpu_info

    def bench(
        self,
        fn: Callable[[], None],
        *,
        name: str = "kernel",
        flops: int = 0,
        bytes_read: int = 0,
        bytes_written: int = 0,
    ) -> ProfileResult:
        """
        Benchmark a GPU kernel function.

        Args:
            fn: Callable that launches the kernel (should not include
                CPU-side data preparation).
            name: Human-readable kernel name for reporting.
            flops: Total floating-point operations per invocation.
            bytes_read: Total bytes read from global memory per invocation.
            bytes_written: Total bytes written to global memory per invocation.

        Returns:
            ProfileResult with timing and derived metrics.
        """
        total_bytes = bytes_read + bytes_written

        # Warmup: also ensures kernel is compiled and cached
        for _ in range(self.warmup):
            fn()
        torch.cuda.synchronize()

        # Benchmark with CUDA events for precise GPU-side timing
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        times_ms: list[float] = []
        for _ in range(self.rep):
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times_ms.append(start.elapsed_time(end))

        # Statistics
        import statistics

        mean_ms = statistics.mean(times_ms)
        std_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0

        result = ProfileResult(name=name, time_ms=mean_ms, time_std_ms=std_ms)

        # TFLOPS: (flops per invocation) / (time in seconds) / 1e12
        if flops > 0:
            result.flops_count = flops
            result.tflops = (flops / (mean_ms * 1e-3)) / 1e12

        # Bandwidth: (bytes per invocation) / (time in seconds) / 1e9
        if total_bytes > 0:
            result.bytes_count = total_bytes
            result.bandwidth_gb_s = (total_bytes / (mean_ms * 1e-3)) / 1e9

        # Roofline bottleneck analysis
        info = self.gpu_info
        if result.tflops > 0 and result.bandwidth_gb_s > 0:
            # Compute the arithmetic intensity (FLOP/byte)
            ai = flops / total_bytes if total_bytes > 0 else float("inf")
            # Ridge point: where compute roof meets memory roof
            ridge_point = info.peak_fp16_tflops / (info.hbm_bandwidth_gb_s / 1000)
            # ^ multiply by 1000 because bandwidth is in GB/s but TFLOPS is T = 1000G

            if ai < ridge_point:
                result.bottleneck = "memory-bound"
            else:
                result.bottleneck = "compute-bound"

        return result

    def print_report(self, *results: ProfileResult) -> None:
        """Print a formatted benchmark report."""
        info = self.gpu_info

        print("=" * 65)
        print(f"  GPU: {info.name}")
        print(f"  Peak fp16: {info.peak_fp16_tflops:.0f} TFLOPS  |  "
              f"Peak fp32: {info.peak_fp32_tflops:.0f} TFLOPS")
        print(f"  HBM bandwidth: {info.hbm_bandwidth_gb_s:.0f} GB/s  |  "
              f"SM count: {info.sm_count}")
        print("=" * 65)

        for r in results:
            print(f"\n{r}")
            if r.tflops > 0 and info.peak_fp16_tflops > 0:
                pct = r.tflops / info.peak_fp16_tflops * 100
                print(f"    Peak %:    {pct:.1f}%")
            if r.bandwidth_gb_s > 0 and info.hbm_bandwidth_gb_s > 0:
                pct = r.bandwidth_gb_s / info.hbm_bandwidth_gb_s * 100
                print(f"    BW util %: {pct:.1f}%")

        print()


# Convenience function for quick benchmarks
def quick_bench(
    fn: Callable[[], None],
    name: str = "kernel",
    warmup: int = 5,
    rep: int = 100,
    flops: int = 0,
    bytes_read: int = 0,
    bytes_written: int = 0,
) -> ProfileResult:
    """One-liner benchmark helper."""
    profiler = KernelProfiler(warmup=warmup, rep=rep)
    result = profiler.bench(
        fn,
        name=name,
        flops=flops,
        bytes_read=bytes_read,
        bytes_written=bytes_written,
    )
    profiler.print_report(result)
    return result


# ---------------------------------------------------------------------------
# Multi-implementation comparison
# ---------------------------------------------------------------------------


@dataclass
class BenchResult:
    """Single implementation benchmark result."""

    name: str
    time_ms: float
    time_std_ms: float = 0.0
    tflops: float = 0.0
    bandwidth_gbs: float = 0.0
    pct_of_ceiling: float = 0.0
    speedup_vs_baseline: float = 1.0


@dataclass
class CompareResult:
    """Multi-implementation comparison result."""

    results: list[BenchResult] = field(default_factory=list)
    ceiling_tflops: float = 0.0
    ceiling_bandwidth_gbs: float = 0.0
    bottleneck: str = ""
    ridge_point: float = 0.0
    arithmetic_intensity: float = 0.0


def bench_compare(
    implementations: dict[str, Callable],
    flops: int = 0,
    bytes_accessed: int = 0,
    dtype: str = "fp16",
    warmup: int = 25,
    rep: int = 100,
    baseline_name: str = "",
) -> CompareResult:
    """
    Benchmark multiple implementations side-by-side and compare.

    All implementations are called with the same inputs (no-argument closures).
    Time is measured via CUDA events; median is reported.

    Args:
        implementations: Dict mapping name → no-arg callable.
        flops: FLOPs per invocation (for TFLOPS calculation).
        bytes_accessed: Bytes per invocation (for bandwidth calculation).
        dtype: "fp16", "bf16", or "fp32" for roofline ceiling.
        warmup: Warmup iterations (not measured).
        rep: Measurement iterations.
        baseline_name: Name of the baseline (first impl if empty).
                       Speedup is reported relative to this.

    Returns:
        CompareResult with per-implementation BenchResult entries,
        roofline ceiling, and bottleneck classification.

    Example::

        results = bench_compare({
            "pytorch":     lambda: torch.matmul(a, b),
            "my_triton":   lambda: my_matmul(a, b),
            "my_triton_v2": lambda: my_matmul_v2(a, b),
        }, flops=2*M*N*K, bytes_accessed=bytes)

        for r in results.results:
            print(f"{r.name}: {r.time_ms:.4f}ms, "
                  f"{r.tflops:.1f} TFLOPS, "
                  f"{r.speedup_vs_baseline:.2f}x vs baseline")
    """
    if not implementations:
        return CompareResult()

    import statistics

    # Auto-detect GPU spec for roofline
    info = GPUInfo.detect()

    # Determine baseline
    names = list(implementations.keys())
    if not baseline_name:
        baseline_name = names[0]
    baseline_time: Optional[float] = None

    results: list[BenchResult] = []

    for name, fn in implementations.items():
        # Warmup
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        # Benchmark
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times: list[float] = []

        for _ in range(rep):
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        median_ms = statistics.median(times)
        std_ms = statistics.stdev(times) if len(times) > 1 else 0.0

        # Track baseline time for speedup
        if name == baseline_name:
            baseline_time = median_ms

        # Compute metrics
        tflops_val = 0.0
        bw_val = 0.0
        if flops > 0 and median_ms > 0:
            tflops_val = (flops / (median_ms * 1e-3)) / 1e12
        total_bytes = bytes_accessed
        if total_bytes > 0 and median_ms > 0:
            bw_val = (total_bytes / (median_ms * 1e-3)) / 1e9

        # % of ceiling
        pct = 0.0
        peak = info.peak_fp16_tflops if dtype in ("fp16", "bf16") else info.peak_fp32_tflops
        if peak > 0 and tflops_val > 0:
            pct = tflops_val / peak * 100

        speedup = 1.0
        if baseline_time and baseline_time > 0:
            speedup = baseline_time / median_ms

        results.append(BenchResult(
            name=name,
            time_ms=median_ms,
            time_std_ms=std_ms,
            tflops=tflops_val,
            bandwidth_gbs=bw_val,
            pct_of_ceiling=pct,
            speedup_vs_baseline=speedup,
        ))

    # Roofline analysis
    peak_tflops = info.peak_fp16_tflops if dtype in ("fp16", "bf16") else info.peak_fp32_tflops
    ridge = peak_tflops / (info.hbm_bandwidth_gb_s / 1000) if info.hbm_bandwidth_gb_s > 0 else 0.0
    ai = flops / bytes_accessed if bytes_accessed > 0 else float("inf")
    bottleneck = "compute_bound" if ai >= ridge else "memory_bound"

    return CompareResult(
        results=results,
        ceiling_tflops=peak_tflops,
        ceiling_bandwidth_gbs=info.hbm_bandwidth_gb_s,
        bottleneck=bottleneck,
        ridge_point=ridge,
        arithmetic_intensity=ai,
    )


def print_compare_report(cr: CompareResult) -> None:
    """Print a formatted multi-implementation comparison table."""
    if not cr.results:
        return

    # Column widths
    name_w = max(max(len(r.name) for r in cr.results), 28)

    print(f"  {'─' * (name_w + 62)}")
    print(f"  {'Implementation':<{name_w}s}  {'Time(ms)':>8s}  "
          f"{'TFLOPS':>8s}  {'BW(GB/s)':>8s}  {'Speedup':>8s}  {'% Ceil':>6s}")
    print(f"  {'─' * name_w}  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 6}")

    for r in cr.results:
        speed_str = f"{r.speedup_vs_baseline:.2f}x" if r.speedup_vs_baseline != 1.0 else "1.00x (base)"
        bw_str = f"{r.bandwidth_gbs:.1f}" if r.bandwidth_gbs > 0 else "--"
        print(f"  {r.name:<{name_w}s}  {r.time_ms:>8.4f}  {r.tflops:>8.1f}  "
              f"{bw_str:>8s}  {speed_str:>8s}  {r.pct_of_ceiling:>5.1f}%")

    print(f"  {'─' * (name_w + 62)}")
    if cr.ceiling_tflops > 0:
        print(f"  Roofline ceiling: {cr.ceiling_tflops:.1f} TFLOPS")
    if cr.ceiling_bandwidth_gbs > 0:
        print(f"  Peak bandwidth:   {cr.ceiling_bandwidth_gbs:.1f} GB/s")
    print(f"  Bottleneck:        {cr.bottleneck}")
    print(f"  Arithmetic intensity: {cr.arithmetic_intensity:.1f} FLOP/byte")
    print(f"  Ridge point:       {cr.ridge_point:.1f} FLOP/byte")

    # Find best
    best = max(cr.results, key=lambda r: r.tflops)
    print(f"  Best: {best.name} at {best.tflops:.1f} TFLOPS ({best.pct_of_ceiling:.1f}% of ceiling)")
    print()
