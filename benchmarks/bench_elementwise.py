#!/usr/bin/env python3
"""
Elementwise & Normalization Kernel Three-Tier Benchmark.

Compares your Triton elementwise/norm kernels against:
  Tier 1 (ceiling):  GPU roofline — HBM bandwidth (these are memory-bound)
  Tier 2 (SotA):     Liger Kernel (if installed) / PyTorch fused ops
  Tier 3 (baseline): PyTorch eager (unfused, multiple kernel launches)

  Your kernels:      vector_add, fused_softmax, fused_relu_bias, layer_norm

Usage:
  python benchmarks/bench_elementwise.py
  python benchmarks/bench_elementwise.py --category norm
  python benchmarks/bench_elementwise.py --quick --save
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.roofline import get_gpu_spec, GPUSpec
from benchmarks.references.liger_ref import (
    get_liger_ln, get_liger_rms_norm,
    get_liger_swiglu, get_liger_geglu, get_liger_softmax,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ElemResult:
    name: str
    time_ms: float
    time_std_ms: float = 0.0
    bandwidth_gbs: float = 0.0
    pct_of_ceiling: float = 0.0  # % of peak bandwidth
    max_diff: float = 0.0
    correct: bool = True
    error_msg: str = ""


@dataclass
class CaseResult:
    case_name: str
    category: str
    size_label: str
    results: list[ElemResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Kernel loader
# ---------------------------------------------------------------------------

def _load_fn(module_path: str, func_name: str) -> Optional[Callable]:
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
        return None


# ---------------------------------------------------------------------------
# Benchmark cases
# ---------------------------------------------------------------------------

def build_cases() -> list[dict]:
    """Build all elementwise/norm benchmark cases."""
    cases: list[dict] = []

    # -- Vector Add --
    fn = _load_fn("phase1_fundamentals/01_vector_add", "vector_add")
    cases.append({
        "name": "Vector Add (f32)",
        "category": "elementwise",
        "triton_fn": fn,
        "ref_fn": lambda x, y: x + y,
        "gen": lambda s: ((torch.rand(s, device="cuda", dtype=torch.float32),
                           torch.rand(s, device="cuda", dtype=torch.float32)), {}),
        "bytes": lambda args: args[0].numel() * 3 * 4,  # x+y read + out write
        "flops": lambda args: args[0].numel(),
        "sizes": [65536, 1048576, 16777216, 67108864],
        "labels": ["64K", "1M", "16M", "64M"],
    })

    # -- Fused Softmax --
    fn = _load_fn("phase1_fundamentals/02_fused_softmax", "fused_softmax")
    liger_fn = get_liger_softmax()

    def gen_softmax(size: int):
        x = torch.randn(1024, size, device="cuda", dtype=torch.float32)
        return (x,), {}

    cases.append({
        "name": "Fused Softmax (1024×N)",
        "category": "reduction",
        "triton_fn": fn,
        "liger_fn": liger_fn,
        "ref_fn": lambda x: torch.softmax(x, dim=-1),
        "gen": gen_softmax,
        "bytes": lambda args: args[0].numel() * 2 * 4,
        "flops": lambda args: args[0].numel() * 5,
        "sizes": [256, 1024, 4096, 16384],
        "labels": ["1024×256", "1024×1K", "1024×4K", "1024×16K"],
    })

    # -- Fused ReLU+Bias --
    fn = _load_fn("phase1_fundamentals/03_fused_relu_bias", "fused_relu_bias")
    cases.append({
        "name": "Fused ReLU+Bias",
        "category": "elementwise",
        "triton_fn": fn,
        "ref_fn": lambda x, b: torch.relu(x + b),
        "gen": lambda s: ((torch.randn(s, device="cuda", dtype=torch.float32),
                           torch.randn(s, device="cuda", dtype=torch.float32)), {}),
        "bytes": lambda args: args[0].numel() * 3 * 4,
        "flops": lambda args: args[0].numel() * 2,
        "sizes": [65536, 1048576, 16777216],
        "labels": ["64K", "1M", "16M"],
    })

    # -- Layer Norm --
    fn = _load_fn("phase1_fundamentals/04_layer_norm", "layer_norm")
    liger_ln = get_liger_ln()

    def gen_ln(size: int):
        x = torch.randn(size, 1024, device="cuda", dtype=torch.float32)
        w = torch.randn(1024, device="cuda", dtype=torch.float32)
        b = torch.randn(1024, device="cuda", dtype=torch.float32)
        return (x, w, b), {"eps": 1e-5}

    def ref_ln(x, w, b, eps=1e-5):
        return torch.nn.functional.layer_norm(x, [x.shape[-1]], w, b, eps=eps)

    cases.append({
        "name": "Layer Norm (N×1024)",
        "category": "normalization",
        "triton_fn": fn,
        "liger_fn": liger_ln,
        "ref_fn": ref_ln,
        "gen": gen_ln,
        "bytes": lambda args: args[0].numel() * 3 * 4 + args[0].shape[-1] * 2 * 4,
        "flops": lambda args: args[0].numel() * 8,
        "sizes": [256, 1024, 4096, 16384],
        "labels": ["256×1K", "1K×1K", "4K×1K", "16K×1K"],
        "rtol": 1e-2, "atol": 1e-2,
    })

    # -- Liger RMSNorm --
    liger_rms = get_liger_rms_norm()
    if liger_rms:

        def ref_rms(x, w, eps=1e-5):
            rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
            return x * rms * w

        cases.append({
            "name": "RMSNorm Liger (N×4096)",
            "category": "normalization",
            "triton_fn": liger_rms,
            "ref_fn": ref_rms,
            "gen": lambda s: ((torch.randn(s, 4096, device="cuda", dtype=torch.float32),
                               torch.randn(4096, device="cuda", dtype=torch.float32)),
                              {"eps": 1e-5}),
            "bytes": lambda args: args[0].numel() * 3 * 4 + args[0].shape[-1] * 4,
            "flops": lambda args: args[0].numel() * 6,
            "sizes": [256, 1024, 4096, 16384],
            "labels": ["256×4K", "1K×4K", "4K×4K", "16K×4K"],
        })

    # -- Liger SwiGLU --
    liger_swi = get_liger_swiglu()
    if liger_swi:

        def ref_swiglu(gate, up):
            return gate * torch.nn.functional.silu(up)

        cases.append({
            "name": "SwiGLU Liger (N×4096)",
            "category": "elementwise",
            "triton_fn": liger_swi,
            "ref_fn": ref_swiglu,
            "gen": lambda s: ((torch.randn(s, 4096, device="cuda", dtype=torch.float32),
                               torch.randn(s, 4096, device="cuda", dtype=torch.float32)), {}),
            "bytes": lambda args: args[0].numel() * 3 * 4,
            "flops": lambda args: args[0].numel() * 8,
            "sizes": [256, 1024, 4096, 16384],
            "labels": ["256×4K", "1K×4K", "4K×4K", "16K×4K"],
        })

    return cases


# ---------------------------------------------------------------------------
# Benchmark engine
# ---------------------------------------------------------------------------


class ElementwiseBenchmark:

    def __init__(self, warmup: int = 25, rep: int = 100, quick: bool = False):
        self.warmup = warmup
        self.rep = rep
        self.quick = quick
        self.gpu_spec = get_gpu_spec()

    def _time_fn(self, fn: Callable, args: tuple, kwargs: dict) -> tuple[float, float]:
        for _ in range(self.warmup):
            fn(*args, **kwargs)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times: list[float] = []

        for _ in range(self.rep):
            start.record()
            fn(*args, **kwargs)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        return statistics.median(times), (statistics.stdev(times) if len(times) > 1 else 0.0)

    def _check(self, fn, ref_fn, args, kwargs, rtol=1e-3, atol=1e-3):
        try:
            actual = fn(*args, **kwargs).float()
            expected = ref_fn(*args, **kwargs).float()
            max_diff = (actual - expected).abs().max().item()
            correct = torch.allclose(actual, expected, rtol=rtol, atol=atol)
            return correct, max_diff, ""
        except Exception as e:
            return False, float("inf"), str(e)

    def run(self, category_filter: Optional[str] = None) -> list[CaseResult]:
        cases = build_cases()
        if category_filter:
            cases = [c for c in cases if c["category"] == category_filter]

        print("=" * 80)
        print("  Elementwise & Normalization Benchmark")
        if self.gpu_spec:
            print(f"  GPU: {self.gpu_spec.name}  |  "
                  f"HBM: {self.gpu_spec.hbm_bandwidth_gb_s:.0f} GB/s  |  "
                  f"Peak fp32: {self.gpu_spec.peak_fp32_tflops:.0f} TFLOPS")
        print("=" * 80)

        all_results: list[CaseResult] = []

        for case in cases:
            print(f"\n── {case['name']} ──")
            rtol = case.get("rtol", 1e-3)
            atol = case.get("atol", 1e-3)

            sizes_to_run = case["sizes"][:2] if self.quick else case["sizes"]
            labels_to_run = case["labels"][:2] if self.quick else case["labels"]

            for size, label in zip(sizes_to_run, labels_to_run):
                cr = CaseResult(case_name=case["name"], category=case["category"],
                                size_label=label)

                (args, kwargs) = case["gen"](size)
                byt = case["bytes"](args)
                peak_bw = self.gpu_spec.hbm_bandwidth_gb_s if self.gpu_spec else 0

                # Test each implementation
                impls = [
                    ("Triton (ours)", case.get("triton_fn")),
                    ("Liger (SotA)", case.get("liger_fn")),
                    ("PyTorch (ref)", case.get("ref_fn")),
                ]

                for name, fn in impls:
                    if fn is None:
                        continue

                    is_ref = (name == "PyTorch (ref)")
                    if not is_ref and case.get("ref_fn"):
                        correct, max_diff, err = self._check(
                            fn, case["ref_fn"], args, kwargs, rtol, atol)
                        if not correct and name != "Liger (SotA)":
                            print(f"  [{label}] {name:<20s}  ❌ FAIL: max_diff={max_diff:.4e}")
                            cr.results.append(ElemResult(
                                name=name, time_ms=0, correct=False,
                                max_diff=max_diff, error_msg=err))
                            continue
                    else:
                        correct, max_diff = True, 0.0

                    try:
                        time_ms, time_std = self._time_fn(fn, args, kwargs)
                    except Exception as e:
                        print(f"  [{label}] {name:<20s}  ❌ ERROR: {e}")
                        continue

                    bw = byt / (time_ms * 1e-3) / 1e9 if time_ms > 0 else 0.0
                    pct = bw / peak_bw * 100 if peak_bw > 0 else 0.0

                    status = "✅" if correct else "⚠️"
                    print(f"  [{label}] {name:<20s}  {time_ms:>8.4f}ms  "
                          f"{bw:>7.1f} GB/s  {pct:>5.1f}% BW  {status}")

                    cr.results.append(ElemResult(
                        name=name, time_ms=time_ms, time_std_ms=time_std,
                        bandwidth_gbs=bw, pct_of_ceiling=pct,
                        max_diff=max_diff, correct=correct))

                all_results.append(cr)

        self.print_summary(all_results)
        return all_results

    def print_summary(self, results: list[CaseResult]):
        if not results:
            return

        print(f"\n{'='*80}")
        print("  SUMMARY — Elementwise & Normalization (all memory-bound)")
        if self.gpu_spec:
            print(f"  Peak HBM bandwidth: {self.gpu_spec.hbm_bandwidth_gb_s:.0f} GB/s")
        print(f"{'='*80}")

        by_case: dict[str, list[CaseResult]] = {}
        for cr in results:
            by_case.setdefault(cr.case_name, []).append(cr)

        for case_name, crs in by_case.items():
            print(f"\n  {case_name}:")
            print(f"  {'Size':>12s}  {'Triton':>20s}  {'Liger':>20s}  {'PyTorch':>20s}")
            print(f"  {'-'*12}  {'-'*20}  {'-'*20}  {'-'*20}")

            for cr in crs:
                triton_r = next((r for r in cr.results if "Triton" in r.name), None)
                liger_r = next((r for r in cr.results if "Liger" in r.name), None)
                torch_r = next((r for r in cr.results if "PyTorch" in r.name), None)

                def fmt(r):
                    if r and r.correct:
                        return f"{r.time_ms:.4f}ms ({r.pct_of_ceiling:.0f}%)"
                    return "N/A"

                print(f"  {cr.size_label:>12s}  {fmt(triton_r):>20s}  "
                      f"{fmt(liger_r):>20s}  {fmt(torch_r):>20s}")

        print(f"\n  💡 All elementwise/norm kernels are memory-bound. "
              f"Optimize for bandwidth utilization, not TFLOPS.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Elementwise/norm benchmark: Triton vs Liger vs PyTorch"
    )
    parser.add_argument("--category", "-c", default=None,
                        choices=["elementwise", "reduction", "normalization"])
    parser.add_argument("--quick", "-q", action="store_true")
    parser.add_argument("--save", "-s", action="store_true")
    parser.add_argument("--output", "-o", default="benchmarks/results")
    args = parser.parse_args()

    bench = ElementwiseBenchmark(
        warmup=5 if args.quick else 25,
        rep=20 if args.quick else 100,
        quick=args.quick,
    )

    results = bench.run(category_filter=args.category)

    if args.save:
        os.makedirs(args.output, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        gpu_name = bench.gpu_spec.name.replace(" ", "_") if bench.gpu_spec else "unknown"
        json_path = os.path.join(args.output, f"elementwise_{timestamp}_{gpu_name}.json")

        json_data = {
            "timestamp": timestamp,
            "gpu": bench.gpu_spec.name if bench.gpu_spec else "unknown",
            "benchmark": "elementwise",
            "results": []
        }
        for cr in results:
            entry = {
                "case": cr.case_name, "category": cr.category,
                "size": cr.size_label, "implementations": {},
            }
            for r in cr.results:
                entry["implementations"][r.name] = {
                    "time_ms": r.time_ms, "bandwidth_gbs": r.bandwidth_gbs,
                    "pct_of_ceiling": r.pct_of_ceiling, "correct": r.correct,
                }
            json_data["results"].append(entry)

        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"\nResults saved to: {json_path}")


if __name__ == "__main__":
    main()
