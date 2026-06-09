#!/usr/bin/env python3
"""
GEMM Three-Tier Benchmark.

Compares your Triton GEMM implementations against:
  Tier 1 (ceiling):  GPU roofline — peak TFLOPS for the current GPU
  Tier 2 (SotA):     cuBLAS (via torch.mm) — the gold standard
  Tier 3 (baseline): PyTorch eager matmul (same cuBLAS, but serves as API baseline)

  Your kernels:      matmul_naive, matmul_tiled, matmul_autotuned

Usage:
  python benchmarks/bench_matmul.py
  python benchmarks/bench_matmul.py --quick          # fast mode: fewer sizes, fewer reps
  python benchmarks/bench_matmul.py --save            # save results to JSON
  python benchmarks/bench_matmul.py --plot            # generate TFLOPS vs size plot
  python benchmarks/bench_matmul.py --profile         # torch.profiler detailed trace
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.roofline import get_gpu_spec, roofline_analysis, GPUSpec
from benchmarks.references.cublas_gemm import get_cublas_gemm


# ---------------------------------------------------------------------------
# Problem sizes to sweep
# ---------------------------------------------------------------------------

# (M, N, K) tuples — typical LLM shapes and square matrices
BENCHMARK_SIZES: list[tuple[int, int, int]] = [
    (256,   256,   256),    # small — launch overhead visible
    (512,   512,   512),
    (1024,  1024,  1024),
    (2048,  2048,  2048),  # typical hidden dim
    (4096,  4096,  4096),  # LLM FFN intermediate size
    (8192,  8192,  8192),
    (4096,  4096,  1024),  # non-square: projection down
    (1024,  8192,  4096),  # backward pass shape
]

QUICK_SIZES: list[tuple[int, int, int]] = [
    (256, 256, 256),
    (1024, 1024, 1024),
    (4096, 4096, 4096),
]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class SingleResult:
    """Result for one implementation at one size."""
    name: str
    time_ms: float
    time_std_ms: float = 0.0
    tflops: float = 0.0
    bandwidth_gbs: float = 0.0
    pct_of_ceiling: float = 0.0
    max_diff: float = 0.0
    correct: bool = True
    error_msg: str = ""


@dataclass
class SizeResult:
    """All implementations at one problem size."""
    M: int
    N: int
    K: int
    label: str
    results: list[SingleResult] = field(default_factory=list)
    roofline: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Kernel loader
# ---------------------------------------------------------------------------

def _load_triton_kernel(module_path: str, func_name: str) -> Optional[Callable]:
    """Load a Triton kernel wrapper from a numeric-prefixed file."""
    import importlib
    try:
        spec = importlib.util.spec_from_file_location(
            module_path.replace("/", "_").replace(".", "_"),
            module_path + ".py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, func_name)
    except Exception as e:
        print(f"  [WARN] Could not load {module_path}.{func_name}: {e}")
        return None


# ---------------------------------------------------------------------------
# Benchmark engine
# ---------------------------------------------------------------------------


class MatmulBenchmark:
    """GEMM benchmark runner with three-tier comparison."""

    def __init__(
        self,
        dtype: torch.dtype = torch.float16,
        warmup: int = 25,
        rep: int = 100,
        quick: bool = False,
        profile: bool = False,
        trace_dir: str = "traces",
    ):
        self.dtype = dtype
        self.warmup = warmup
        self.rep = rep
        self.quick = quick
        self.profile = profile
        self.trace_dir = trace_dir
        self.sizes = QUICK_SIZES if quick else BENCHMARK_SIZES
        self.gpu_spec = get_gpu_spec()
        self.results: list[SizeResult] = []

    def _time_fn(self, fn: Callable, args: tuple, kwargs: dict) -> tuple[float, float]:
        """CUDA event timing. Returns (median_ms, std_ms)."""
        # Warmup
        for _ in range(self.warmup):
            fn(*args, **kwargs)
        torch.cuda.synchronize()

        # Benchmark
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times: list[float] = []

        for _ in range(self.rep):
            start.record()
            fn(*args, **kwargs)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        median = statistics.median(times)
        std = statistics.stdev(times) if len(times) > 1 else 0.0
        return median, std

    def _check_correctness(
        self, fn: Callable, ref_fn: Callable, args: tuple, kwargs: dict,
        rtol: float = 0.01, atol: float = 0.01,
    ) -> tuple[bool, float, str]:
        """Verify Triton output vs reference."""
        try:
            actual = fn(*args, **kwargs).float()
            expected = ref_fn(*args, **kwargs).float()
            max_diff = (actual - expected).abs().max().item()
            correct = torch.allclose(actual, expected, rtol=rtol, atol=atol)
            return correct, max_diff, ""
        except Exception as e:
            return False, float("inf"), str(e)

    def _compute_flops(self, M: int, N: int, K: int) -> int:
        return 2 * M * N * K

    def _compute_bytes(self, M: int, N: int, K: int) -> int:
        dtype_bytes = 2 if self.dtype == torch.float16 else 4
        return (M * K + K * N + M * N) * dtype_bytes

    def _profile_fn(self, fn: Callable, args: tuple, kwargs: dict, label: str):
        """Run with torch.profiler for detailed trace."""
        print(f"\n  [torch.profiler] {label}")
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
        ) as prof:
            for _ in range(3):
                fn(*args, **kwargs)
            torch.cuda.synchronize()

        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=8))

        if self.trace_dir:
            os.makedirs(self.trace_dir, exist_ok=True)
            trace_path = os.path.join(self.trace_dir, f"matmul_{label}.json")
            prof.export_chrome_trace(trace_path)
            print(f"  Trace saved: {trace_path}")

    # ------------------------------------------------------------------
    # Load all available implementations
    # ------------------------------------------------------------------

    def _get_implementations(self) -> dict[str, Optional[Callable]]:
        """Discover all available GEMM implementations."""
        impls: dict[str, Optional[Callable]] = {}

        # Tier 2: cuBLAS (always available)
        impls["cuBLAS (torch.mm)"] = get_cublas_gemm(self.dtype)

        # Your Triton kernels
        impls["matmul_naive (Triton)"] = _load_triton_kernel(
            "phase2_compute/01_matmul_naive", "matmul_naive")
        impls["matmul_tiled (Triton)"] = _load_triton_kernel(
            "phase2_compute/02_matmul_tiled", "matmul_tiled")
        impls["matmul_autotuned (Triton)"] = _load_triton_kernel(
            "phase2_compute/03_matmul_autotuned", "matmul_autotuned")

        # Also load from benchmark_triton.py (simple standalone kernels)
        impls["benchmark_triton matmul"] = _load_triton_kernel(
            "benchmark_triton", "matmul_triton")

        return impls

    # ------------------------------------------------------------------
    # Run benchmark
    # ------------------------------------------------------------------

    def run(self, kernel_filter: Optional[str] = None) -> list[SizeResult]:
        """Run full GEMM benchmark across all sizes."""
        impls = self._get_implementations()

        # Filter
        if kernel_filter:
            impls = {k: v for k, v in impls.items()
                     if kernel_filter.lower() in k.lower()}

        # Print header
        print("=" * 80)
        print("  GEMM Benchmark — Triton vs cuBLAS vs Roofline")
        if self.gpu_spec:
            print(f"  GPU: {self.gpu_spec.name}  |  "
                  f"Peak fp16: {self.gpu_spec.peak_fp16_tflops:.0f} TFLOPS  |  "
                  f"HBM: {self.gpu_spec.hbm_bandwidth_gb_s:.0f} GB/s")
        print(f"  dtype: {self.dtype}  |  warmup={self.warmup}  |  rep={self.rep}")
        print("=" * 80)

        all_results: list[SizeResult] = []

        # cuBLAS is the reference for correctness
        ref_fn = impls.get("cuBLAS (torch.mm)")

        for M, N, K in self.sizes:
            label = f"{M}×{N}×{K}"
            print(f"\n── {label} ──")

            size_result = SizeResult(M=M, N=N, K=K, label=label)

            # Generate inputs
            a = torch.randn(M, K, device="cuda", dtype=self.dtype)
            b = torch.randn(K, N, device="cuda", dtype=self.dtype)
            flops = self._compute_flops(M, N, K)
            byt = self._compute_bytes(M, N, K)

            # Roofline ceiling
            if self.gpu_spec:
                size_result.roofline = {
                    "ceiling_tflops": self.gpu_spec.peak_fp16_tflops,
                    "arithmetic_intensity": flops / byt if byt > 0 else float("inf"),
                    "ridge_point": self.gpu_spec.ridge_point_fp16,
                    "bottleneck": ("compute_bound" if flops / byt >= self.gpu_spec.ridge_point_fp16
                                   else "memory_bound") if byt > 0 else "unknown",
                }

            for name, fn in impls.items():
                if fn is None:
                    print(f"  {name:<30s}  (not available)")
                    continue

                # Correctness vs cuBLAS
                is_ref = (name == "cuBLAS (torch.mm)")
                if not is_ref and ref_fn:
                    correct, max_diff, err = self._check_correctness(
                        fn, ref_fn, (a, b), {})
                    if not correct:
                        print(f"  {name:<30s}  ❌ FAIL: {err or f'max_diff={max_diff:.4e}'}")
                        size_result.results.append(SingleResult(
                            name=name, time_ms=0, correct=False,
                            max_diff=max_diff, error_msg=err))
                        continue
                else:
                    correct, max_diff = True, 0.0

                # Timing
                try:
                    time_ms, time_std = self._time_fn(fn, (a, b), {})
                except Exception as e:
                    print(f"  {name:<30s}  ❌ ERROR: {e}")
                    size_result.results.append(SingleResult(
                        name=name, time_ms=0, correct=False, error_msg=str(e)))
                    continue

                # Metrics
                tflops = flops / (time_ms * 1e-3) / 1e12 if time_ms > 0 else 0.0
                bw = byt / (time_ms * 1e-3) / 1e9 if time_ms > 0 else 0.0

                pct_ceiling = 0.0
                if self.gpu_spec:
                    pct_ceiling = tflops / self.gpu_spec.peak_fp16_tflops * 100

                status = "✅" if correct else "❌"
                print(f"  {name:<30s}  {time_ms:>8.4f}ms ±{time_std:.4f}  "
                      f"{tflops:>7.1f} TFLOPS  {pct_ceiling:>5.1f}% ceil  {status}")

                size_result.results.append(SingleResult(
                    name=name, time_ms=time_ms, time_std_ms=time_std,
                    tflops=tflops, bandwidth_gbs=bw,
                    pct_of_ceiling=pct_ceiling,
                    max_diff=max_diff, correct=correct))

            all_results.append(size_result)

        self.results = all_results

        # Profile if requested
        if self.profile:
            self._run_profiles()

        # Print summary
        self.print_summary()

        return all_results

    def _run_profiles(self):
        """Profile middle size with torch.profiler."""
        if not self.sizes:
            return
        mid_size = self.sizes[len(self.sizes) // 2]
        M, N, K = mid_size
        a = torch.randn(M, K, device="cuda", dtype=self.dtype)
        b = torch.randn(K, N, device="cuda", dtype=self.dtype)

        print(f"\n{'='*80}")
        print("  torch.profiler — Detailed traces")
        print(f"{'='*80}")

        # Profile Triton tiled
        tiled_fn = _load_triton_kernel("phase2_compute/02_matmul_tiled", "matmul_tiled")
        if tiled_fn:
            self._profile_fn(tiled_fn, (a, b), {}, f"triton_tiled_{M}x{N}x{K}")

        # Profile cuBLAS
        cublas_fn = get_cublas_gemm(self.dtype)
        if cublas_fn:
            self._profile_fn(cublas_fn, (a, b), {}, f"cublas_{M}x{N}x{K}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------

    def print_summary(self):
        """Print a comprehensive comparison table."""
        if not self.results:
            return

        print(f"\n{'='*100}")
        print("  SUMMARY — GEMM Performance Comparison")
        print(f"{'='*100}")

        # Collect across all sizes
        print(f"  {'Size':>12s}  ", end="")
        # Get all unique names
        all_names: list[str] = []
        for sr in self.results:
            for r in sr.results:
                if r.name not in all_names:
                    all_names.append(r.name)

        for name in all_names:
            print(f"{name[:20]:>20s}  ", end="")
        print(f"{'%cuBLAS':>8s}  {'%Ceiling':>8s}")
        print(f"  {'-'*12}  " + "  ".join(f"{'-'*20}" for _ in all_names) + f"  {'-'*8}  {'-'*8}")

        cublas_name = "cuBLAS (torch.mm)"

        for sr in self.results:
            print(f"  {sr.label:>12s}  ", end="")

            cublas_tflops = 1.0
            # Find cuBLAS result for this size
            for r in sr.results:
                if r.name == cublas_name:
                    cublas_tflops = r.tflops if r.tflops > 0 else 1.0
                    break

            best_pct_ceiling = 0.0
            for name in all_names:
                found = [r for r in sr.results if r.name == name]
                if found and found[0].correct:
                    r = found[0]
                    pct_cublas = r.tflops / cublas_tflops * 100 if cublas_tflops > 0 else 0
                    print(f"{r.tflops:>7.1f} TF ({pct_cublas:>4.0f}%)  ", end="")
                    if r.pct_of_ceiling > best_pct_ceiling:
                        best_pct_ceiling = r.pct_of_ceiling
                else:
                    print(f"{'N/A':>20s}  ", end="")

            # % of ceiling for best implementation
            print(f"{'':>8s}  {best_pct_ceiling:>5.1f}%")

        # Legend
        print(f"\n  Roofline ceiling: {self.gpu_spec.peak_fp16_tflops:.0f} TFLOPS (fp16)"
              if self.gpu_spec else "")
        if self.gpu_spec:
            print(f"  Ridge point: {self.gpu_spec.ridge_point_fp16:.1f} FLOP/byte")

    # ------------------------------------------------------------------
    # Plot (matplotlib)
    # ------------------------------------------------------------------

    def plot(self, save_path: Optional[str] = None):
        """Generate TFLOPS vs matrix size plot."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[WARN] matplotlib not installed — skipping plot.")
            return

        if not self.results:
            print("No results to plot.")
            return

        # Group by implementation name
        by_name: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for sr in self.results:
            size_label = sr.M  # use M as proxy for "size"
            for r in sr.results:
                if r.correct:
                    by_name[r.name].append((float(size_label), r.tflops))

        fig, ax = plt.subplots(figsize=(10, 6))

        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
        for i, (name, points) in enumerate(sorted(by_name.items())):
            if not points:
                continue
            xs, ys = zip(*points)
            ax.plot(xs, ys, marker=markers[i % len(markers)], label=name, linewidth=2)

        # Roofline ceiling (dashed)
        if self.gpu_spec:
            x_range = [s[0] for s in self.sizes]
            ceiling = [self.gpu_spec.peak_fp16_tflops] * len(x_range)
            ax.plot(x_range, ceiling, 'k--', linewidth=1,
                    label=f"Peak fp16 ({self.gpu_spec.peak_fp16_tflops:.0f} TFLOPS)")

        ax.set_xlabel("Matrix dimension (M = N = K)")
        ax.set_ylabel("TFLOPS")
        ax.set_title(f"GEMM Performance — {self.gpu_spec.name if self.gpu_spec else 'GPU'}")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log", base=2)

        # Secondary axis: % of ceiling
        if self.gpu_spec:
            ax2 = ax.twinx()
            y_max = ax.get_ylim()[1]
            ax2.set_ylim(0, y_max / self.gpu_spec.peak_fp16_tflops * 100)
            ax2.set_ylabel("% of Peak TFLOPS")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="GEMM benchmark: Triton vs cuBLAS vs roofline"
    )
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--quick", "-q", action="store_true",
                        help="Fast mode: fewer sizes and reps")
    parser.add_argument("--save", "-s", action="store_true",
                        help="Save results to JSON")
    parser.add_argument("--plot", "-p", action="store_true",
                        help="Generate TFLOPS vs size plot")
    parser.add_argument("--profile", action="store_true",
                        help="Run torch.profiler on representative sizes")
    parser.add_argument("--trace-dir", default="traces",
                        help="Directory for chrome traces")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory for results")
    parser.add_argument("--kernel", "-k", default=None,
                        help="Filter to specific kernel (substring)")
    args = parser.parse_args()

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    bench = MatmulBenchmark(
        dtype=dtype,
        warmup=5 if args.quick else 25,
        rep=20 if args.quick else 100,
        quick=args.quick,
        profile=args.profile,
        trace_dir=args.trace_dir,
    )

    results = bench.run(kernel_filter=args.kernel)

    # Save results
    if args.save or args.output:
        out_dir = args.output or "benchmarks/results"
        os.makedirs(out_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        gpu_name = bench.gpu_spec.name.replace(" ", "_") if bench.gpu_spec else "unknown"
        json_path = os.path.join(out_dir, f"matmul_{timestamp}_{gpu_name}.json")

        json_data = {
            "timestamp": timestamp,
            "gpu": bench.gpu_spec.name if bench.gpu_spec else "unknown",
            "benchmark": "matmul",
            "dtype": args.dtype,
            "results": []
        }
        for sr in results:
            entry = {
                "M": sr.M, "N": sr.N, "K": sr.K,
                "label": sr.label,
                "implementations": {},
                "roofline": sr.roofline,
            }
            for r in sr.results:
                entry["implementations"][r.name] = {
                    "time_ms": r.time_ms, "time_std_ms": r.time_std_ms,
                    "tflops": r.tflops, "pct_of_ceiling": r.pct_of_ceiling,
                    "correct": r.correct,
                }
            json_data["results"].append(entry)

        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"\nResults saved to: {json_path}")

    # Plot
    if args.plot:
        plot_path = None
        if args.output:
            plot_path = os.path.join(args.output, "matmul_tflops.png")
        bench.plot(save_path=plot_path)


if __name__ == "__main__":
    main()
