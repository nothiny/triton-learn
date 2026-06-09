#!/usr/bin/env python3
"""
Benchmark Runner — Compare Triton kernels against PyTorch/cuBLAS.

Features:
  - Auto-discovers all Triton kernels from phase1/phase2
  - Compares against PyTorch reference (cuBLAS/cuDNN)
  - Computes TFLOPS, bandwidth, roofline bottleneck
  - torch.profiler integration for detailed timeline
  - Side-by-side comparison tables
  - Optional: generate chrome trace for visualization

Usage:
  # Quick comparison
  python benchmarks/bench_runner.py

  # With torch profiler
  python benchmarks/bench_runner.py --profile

  # Specific category only
  python benchmarks/bench_runner.py --category gemm

  # Export chrome trace
  python benchmarks/bench_runner.py --profile --trace-out traces/

  # Single kernel
  python benchmarks/bench_runner.py --kernel "MatMul Tiled"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.profiler import GPUInfo, KernelProfiler, ProfileResult
from benchmarks.bench_cases import BenchCase, build_cases


# ---------------------------------------------------------------------------
# Extended profile result with comparison
# ---------------------------------------------------------------------------


@dataclass
class ComparisonResult:
    """Result of comparing Triton kernel vs reference."""

    case: BenchCase
    size_label: str
    triton: Optional[ProfileResult]  # None if kernel failed
    reference: ProfileResult
    speedup: float  # triton_time / ref_time (>1 means triton is slower)
    tflops_pct_of_peak: float = 0.0
    max_diff: float = 0.0
    correct: bool = True
    error_msg: str = ""

    @property
    def triton_ms(self) -> Optional[float]:
        return self.triton.time_ms if self.triton else None

    @property
    def ref_ms(self) -> float:
        return self.reference.time_ms


# ---------------------------------------------------------------------------
# Benchmark Engine
# ---------------------------------------------------------------------------


class BenchRunner:
    """Main benchmark runner."""

    def __init__(
        self,
        profile: bool = False,
        trace_out: Optional[str] = None,
        gpu_info: Optional[GPUInfo] = None,
    ):
        self.profile_flag = profile
        self.trace_out = trace_out
        self.gpu_info = gpu_info or GPUInfo.detect()
        self.profiler = KernelProfiler(warmup=5, rep=50)
        self.results: list[ComparisonResult] = []

    # ------------------------------------------------------------------
    # Timing helpers
    # ------------------------------------------------------------------

    def _time_fn(
        self, fn: Callable, args: tuple, kwargs: dict, warmup: int, rep: int
    ) -> tuple[float, float]:
        """Time a function using CUDA events. Returns (mean_ms, std_ms)."""
        # Warmup
        for _ in range(warmup):
            fn(*args, **kwargs)
        torch.cuda.synchronize()

        # Benchmark
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times: list[float] = []

        for _ in range(rep):
            start.record()
            fn(*args, **kwargs)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        import statistics
        mean = statistics.mean(times)
        std = statistics.stdev(times) if len(times) > 1 else 0.0
        return mean, std

    # ------------------------------------------------------------------
    # Profile with torch.profiler
    # ------------------------------------------------------------------

    def _profile_fn(
        self, fn: Callable, args: tuple, kwargs: dict, label: str
    ) -> None:
        """Run with torch.profiler and print kernel breakdown."""
        print(f"\n  ── torch.profiler: {label} ──")

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=True,
            with_modules=True,
        ) as prof:
            for _ in range(3):  # Fewer iterations for profiling
                fn(*args, **kwargs)
            torch.cuda.synchronize()

        # Print key table
        print(prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=10
        ))

        # Export chrome trace
        if self.trace_out:
            os.makedirs(self.trace_out, exist_ok=True)
            safe_name = label.replace(" ", "_").replace("/", "_")
            trace_path = os.path.join(self.trace_out, f"{safe_name}.json")
            prof.export_chrome_trace(trace_path)
            print(f"  Chrome trace saved to: {trace_path}")

    # ------------------------------------------------------------------
    # Correctness check
    # ------------------------------------------------------------------

    def _check_correctness(
        self, case: BenchCase, args: tuple, kwargs: dict
    ) -> tuple[bool, float, str]:
        """Check Triton output matches reference."""
        if case.triton_fn is None:
            return False, float("inf"), "Triton kernel not available"

        try:
            actual = case.triton_fn(*args, **kwargs)
            expected = case.ref_fn(*args, **kwargs)

            # Convert to float32 for comparison
            actual_f32 = actual.detach().float()
            expected_f32 = expected.detach().float()

            max_diff = (actual_f32 - expected_f32).abs().max().item()
            passed = torch.allclose(actual_f32, expected_f32,
                                     rtol=case.rtol, atol=case.atol)
            return passed, max_diff, ""
        except Exception as e:
            return False, float("inf"), str(e)

    # ------------------------------------------------------------------
    # Run single case
    # ------------------------------------------------------------------

    def run_case(self, case: BenchCase) -> list[ComparisonResult]:
        """Run one benchmark case across all sizes."""
        results: list[ComparisonResult] = []

        print(f"\n{'='*70}")
        print(f"  {case.name}")
        print(f"{'='*70}")

        for size, label in zip(case.sizes, case.size_labels):
            print(f"\n  [{label}]", end=" ", flush=True)

            # Generate inputs
            (args, kwargs) = case.input_gen(size)

            # Step 1: Correctness check
            if case.triton_fn:
                ok, max_diff, err = self._check_correctness(case, args, kwargs)
                if not ok:
                    print(f"❌ correctness FAIL: {err or f'max_diff={max_diff:.4e}'}")
                    results.append(ComparisonResult(
                        case=case, size_label=label,
                        triton=None,
                        reference=ProfileResult(name="ref", time_ms=0),
                        speedup=float("inf"), max_diff=max_diff,
                        correct=False, error_msg=err,
                    ))
                    continue
                print(f"✅ max_diff={max_diff:.2e}", end=" ", flush=True)
            else:
                max_diff = 0.0
                print("⚠️  skip (no kernel)", end=" ", flush=True)

            # Step 2: Time reference
            warmup = case.warmup
            rep = case.rep
            ref_ms, ref_std = self._time_fn(case.ref_fn, args, kwargs, warmup, rep)

            # Step 3: Time Triton
            triton_ms = None
            triton_std = None
            if case.triton_fn:
                triton_ms, triton_std = self._time_fn(
                    case.triton_fn, args, kwargs, warmup, rep
                )
                speedup = triton_ms / ref_ms if ref_ms > 0 else float("inf")
            else:
                speedup = float("inf")

            # Step 4: Compute metrics
            flops = case.flops_calc(args)
            byt = case.bytes_calc(args)

            triton_result = None
            ref_result = ProfileResult(
                name=case.name,
                time_ms=ref_ms,
                time_std_ms=ref_std,
                flops_count=flops,
                bytes_count=byt,
            )
            # Fill derived metrics
            if flops > 0:
                ref_result.tflops = (flops / (ref_ms * 1e-3)) / 1e12
            if byt > 0:
                ref_result.bandwidth_gb_s = (byt / (ref_ms * 1e-3)) / 1e9

            if triton_ms is not None:
                triton_result = ProfileResult(
                    name=f"Triton-{case.name}",
                    time_ms=triton_ms,
                    time_std_ms=triton_std or 0,
                    flops_count=flops,
                    bytes_count=byt,
                )
                if flops > 0:
                    triton_result.tflops = (flops / (triton_ms * 1e-3)) / 1e12
                if byt > 0:
                    triton_result.bandwidth_gb_s = (byt / (triton_ms * 1e-3)) / 1e9

            print(f"| Triton: {triton_ms:.4f}ms | Ref: {ref_ms:.4f}ms | "
                  f"Speedup: {speedup:.2f}x", flush=True)

            # Print perf details
            if triton_result:
                print(f"         Triton TFLOPS={triton_result.tflops:.1f}  "
                      f"BW={triton_result.bandwidth_gb_s:.1f} GB/s")
            print(f"         Ref     TFLOPS={ref_result.tflops:.1f}  "
                  f"BW={ref_result.bandwidth_gb_s:.1f} GB/s")

            cr = ComparisonResult(
                case=case,
                size_label=label,
                triton=triton_result,
                reference=ref_result,
                speedup=speedup,
                max_diff=max_diff,
                correct=True,
            )
            results.append(cr)

        return results

    # ------------------------------------------------------------------
    # Profile single case (with torch.profiler)
    # ------------------------------------------------------------------

    def profile_case(self, case: BenchCase, size_idx: int = -1):
        """Run torch.profiler on a specific case+size combination."""
        size_idx = size_idx % len(case.sizes)
        size = case.sizes[size_idx]
        label = case.size_labels[size_idx]
        (args, kwargs) = case.input_gen(size)

        print(f"\n{'='*70}")
        print(f"  PROFILE: {case.name}  [{label}]")
        print(f"{'='*70}")

        # Profile Triton
        if case.triton_fn:
            self._profile_fn(case.triton_fn, args, kwargs,
                             f"{case.name}_Triton_{label}")

        # Profile reference
        self._profile_fn(case.ref_fn, args, kwargs,
                         f"{case.name}_Ref_{label}")

    # ------------------------------------------------------------------
    # Full run
    # ------------------------------------------------------------------

    def run_all(
        self,
        cases: list[BenchCase],
        category_filter: Optional[str] = None,
    ):
        """Run all benchmarks."""
        if category_filter:
            cases = [c for c in cases if c.category == category_filter]
            print(f"Filtered to category: {category_filter} ({len(cases)} cases)")

        print(f"\n{'#'*70}")
        print(f"  Triton Kernel Benchmark Suite")
        print(f"  GPU: {self.gpu_info.name}")
        print(f"  Peak fp16: {self.gpu_info.peak_fp16_tflops:.0f} TFLOPS | "
              f"fp32: {self.gpu_info.peak_fp32_tflops:.0f} TFLOPS")
        print(f"  HBM Bandwidth: {self.gpu_info.hbm_bandwidth_gb_s:.0f} GB/s")
        print(f"{'#'*70}")

        for case in cases:
            results = self.run_case(case)
            self.results.extend(results)

        # Summary
        self.print_summary()

        # Profile selected cases if requested
        if self.profile_flag:
            self._run_profiles(cases)

    def _run_profiles(self, cases: list[BenchCase]):
        """Run torch.profiler on a representative subset."""
        print(f"\n{'#'*70}")
        print(f"  Detailed Profiling (torch.profiler)")
        print(f"{'#'*70}")

        for case in cases:
            # Profile the middle size for each case
            mid_idx = len(case.sizes) // 2
            self.profile_case(case, mid_idx)

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------

    def print_summary(self):
        """Print a comprehensive comparison table."""
        if not self.results:
            return

        print(f"\n{'='*100}")
        print(f"  SUMMARY: Triton vs PyTorch/cuBLAS Comparison")
        print(f"{'='*100}")

        # Group by category
        from collections import defaultdict
        by_category = defaultdict(list)
        for r in self.results:
            by_category[r.case.category].append(r)

        for category, results in sorted(by_category.items()):
            print(f"\n  ── {category.upper()} ──")
            print(f"  {'Kernel':<35s} {'Size':>10s}  {'Triton(ms)':>10s}  "
                  f"{'Ref(ms)':>10s}  {'Speedup':>8s}  {'Triton TFLOPS':>13s}  "
                  f"{'Ref TFLOPS':>12s}  {'Correct':>8s}")
            print(f"  {'-'*35} {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}  "
                  f"{'-'*13}  {'-'*12}  {'-'*8}")

            for r in results:
                triton_ms_str = f"{r.triton_ms:.4f}" if r.triton_ms else "N/A"
                triton_tf = f"{r.triton.tflops:.1f}" if r.triton and r.triton.tflops > 0 else "-"
                ref_tf = f"{r.reference.tflops:.1f}" if r.reference.tflops > 0 else "-"

                # Speedup interpretation: >1 means Triton is slower, <1 means faster
                if r.triton_ms is None:
                    speed_str = "FAIL"
                elif r.speedup < 1.0:
                    speed_str = f"🔥 {r.speedup:.2f}x"
                elif r.speedup < 1.5:
                    speed_str = f"✅ {r.speedup:.2f}x"
                elif r.speedup < 5.0:
                    speed_str = f"⚠️  {r.speedup:.2f}x"
                else:
                    speed_str = f"❌ {r.speedup:.2f}x"

                correct_str = "✅" if r.correct else "❌"

                # Truncate case name for table
                name = r.case.name[:34]

                print(f"  {name:<35s} {r.size_label:>10s}  {triton_ms_str:>10s}  "
                      f"{r.ref_ms:>10.4f}  {speed_str:>8s}  {triton_tf:>13s}  "
                      f"{ref_tf:>12s}  {correct_str:>8s}")

        # Overall summary
        n_pass = sum(1 for r in self.results if r.correct)
        n_total = len(self.results)
        faster = sum(1 for r in self.results if r.triton_ms and r.speedup < 1.0)
        slower = sum(1 for r in self.results if r.triton_ms and r.speedup >= 1.0)

        print(f"\n  {'─'*90}")
        print(f"  Total: {n_total} benchmarks | ✅ {n_pass} correct | "
              f"🔥 {faster} faster than ref | ⚠️  {slower} slower than ref")
        print(f"  Speedup legend: 🔥<1.0x (faster) | ✅1.0-1.5x | ⚠️ 1.5-5.0x | ❌>5.0x")

        # Roofline insight
        print(f"\n  ── Roofline Context ──")
        print(f"  GPU: {self.gpu_info.name}")
        print(f"  Peak fp16 Tensor Core: {self.gpu_info.peak_fp16_tflops:.0f} TFLOPS")
        print(f"  Peak fp32: {self.gpu_info.peak_fp32_tflops:.0f} TFLOPS")
        print(f"  HBM Bandwidth: {self.gpu_info.hbm_bandwidth_gb_s:.0f} GB/s")
        print(f"  Ridge Point: {self.gpu_info.peak_fp16_tflops / self.gpu_info.hbm_bandwidth_gb_s * 1000:.0f} FLOP/byte")
        print(f"  (Kernels with arithmetic intensity below ridge point are memory-bound)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Triton kernels vs PyTorch/cuBLAS"
    )
    parser.add_argument(
        "--category", "-c", type=str, default=None,
        choices=["elementwise", "reduction", "gemm", "attention", "normalization"],
        help="Only run benchmarks in this category"
    )
    parser.add_argument(
        "--kernel", "-k", type=str, default=None,
        help="Only run benchmark for this kernel (substring match)"
    )
    parser.add_argument(
        "--profile", "-p", action="store_true",
        help="Run torch.profiler on each case (slow)"
    )
    parser.add_argument(
        "--trace-out", "-t", type=str, default="traces",
        help="Directory for chrome trace exports (default: traces/)"
    )
    parser.add_argument(
        "--quick", "-q", action="store_true",
        help="Quick mode: fewer iterations, smaller sizes"
    )
    parser.add_argument(
        "--json", "-j", type=str, default=None,
        help="Export results to JSON file"
    )
    args = parser.parse_args()

    # Load cases
    all_cases = build_cases()

    # Filter
    if args.category:
        all_cases = [c for c in all_cases if c.category == args.category]
    if args.kernel:
        all_cases = [c for c in all_cases if args.kernel.lower() in c.name.lower()]

    if not all_cases:
        print("No benchmark cases match filters.")
        return

    # Quick mode
    if args.quick:
        for c in all_cases:
            c.warmup = 3
            c.rep = 20
            c.sizes = c.sizes[:3]  # only smaller sizes

    # Run
    runner = BenchRunner(
        profile=args.profile,
        trace_out=args.trace_out,
    )
    runner.run_all(all_cases, category_filter=None)

    # Export JSON
    if args.json:
        export_json(runner.results, args.json)


def export_json(results: list[ComparisonResult], path: str):
    """Export results as JSON for further analysis."""
    data = []
    for r in results:
        data.append({
            "kernel": r.case.name,
            "category": r.case.category,
            "size": r.size_label,
            "triton_ms": r.triton_ms,
            "ref_ms": r.ref_ms,
            "speedup": r.speedup,
            "triton_tflops": r.triton.tflops if r.triton else None,
            "ref_tflops": r.reference.tflops,
            "max_diff": r.max_diff,
            "correct": r.correct,
        })

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults exported to: {path}")


if __name__ == "__main__":
    main()
