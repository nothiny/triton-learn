#!/usr/bin/env python3
"""
Attention Kernel Three-Tier Benchmark.

Compares your Triton Flash Attention implementations against:
  Tier 1 (ceiling):   GPU roofline — memory-bound for long sequences
  Tier 2 (SotA):      flash-attn library (Tri Dao) / torch SDPA
  Tier 3 (baseline):  Naive attention (O(N²) memory, three separate kernels)

  Your kernels:       flash_attention_v1, flash_attention_v2

Sweeps over sequence lengths at fixed batch/heads/head_dim (GPT-2 large config).

Usage:
  python benchmarks/bench_attention.py
  python benchmarks/bench_attention.py --quick
  python benchmarks/bench_attention.py --save --plot
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

from utils.roofline import get_gpu_spec, GPUSpec
from benchmarks.references.flash_attn_ref import (
    get_flash_attn, get_torch_sdpa, get_naive_attention,
    compute_attention_flops, compute_attention_bytes,
)


# ---------------------------------------------------------------------------
# Problem sizes (GPT-2 large configuration: heads=32, head_dim=64)
# ---------------------------------------------------------------------------

# Sweep sequence lengths at fixed batch/heads/head_dim
BENCHMARK_CONFIGS: list[dict] = [
    {"batch": 2, "heads": 32, "seq_len": s, "head_dim": 64}
    for s in [128, 256, 512, 1024, 2048, 4096, 8192]
]

QUICK_CONFIGS: list[dict] = [
    {"batch": 2, "heads": 32, "seq_len": s, "head_dim": 64}
    for s in [256, 1024, 4096]
]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class AttentionResult:
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
    batch: int
    heads: int
    seq_len: int
    head_dim: int
    label: str
    results: list[AttentionResult] = field(default_factory=list)
    memory: dict = field(default_factory=dict)
    roofline: dict = field(default_factory=dict)


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
        print(f"  [WARN] Could not load {module_path}.{func_name}: {e}")
        return None


# ---------------------------------------------------------------------------
# Benchmark engine
# ---------------------------------------------------------------------------


class AttentionBenchmark:

    def __init__(
        self,
        dtype: torch.dtype = torch.float16,
        warmup: int = 10,
        rep: int = 50,
        quick: bool = False,
    ):
        self.dtype = dtype
        self.warmup = warmup
        self.rep = rep
        self.quick = quick
        self.configs = QUICK_CONFIGS if quick else BENCHMARK_CONFIGS
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

    def _check(self, fn: Callable, ref_fn: Callable, args: tuple, kwargs: dict,
               rtol: float = 0.1, atol: float = 0.1) -> tuple[bool, float, str]:
        try:
            actual = fn(*args, **kwargs).float()
            expected = ref_fn(*args, **kwargs).float()
            max_diff = (actual - expected).abs().max().item()
            correct = torch.allclose(actual, expected, rtol=rtol, atol=atol)
            return correct, max_diff, ""
        except Exception as e:
            return False, float("inf"), str(e)

    def _get_implementations(self) -> dict[str, Optional[Callable]]:
        impls: dict[str, Optional[Callable]] = {}

        # Tier 2: SotA references
        impls["flash-attn (Tri Dao)"] = get_flash_attn()
        impls["torch SDPA (FlashAttn)"] = get_torch_sdpa()

        # Tier 3: Naive baseline
        impls["naive attention"] = get_naive_attention()

        # Your Triton kernels
        impls["flash_attn_v1 (Triton)"] = _load_fn(
            "phase2_compute/07_flash_attention_v1", "flash_attention_v1")
        impls["flash_attn_v2 (Triton)"] = _load_fn(
            "phase2_compute/08_flash_attention_v2", "flash_attention_v2")

        return impls

    def run(self, kernel_filter: Optional[str] = None) -> list[SizeResult]:
        impls = self._get_implementations()

        if kernel_filter:
            impls = {k: v for k, v in impls.items()
                     if kernel_filter.lower() in k.lower()}

        print("=" * 80)
        print("  Attention Benchmark — Flash Attention vs SDPA vs Naive")
        if self.gpu_spec:
            print(f"  GPU: {self.gpu_spec.name}  |  "
                  f"Peak fp16: {self.gpu_spec.peak_fp16_tflops:.0f} TFLOPS  |  "
                  f"HBM: {self.gpu_spec.hbm_bandwidth_gb_s:.0f} GB/s")
        print("=" * 80)

        all_results: list[SizeResult] = []
        ref_fn = impls.get("torch SDPA (FlashAttn)") or impls.get("naive attention")

        for cfg in self.configs:
            B, H, N, D = cfg["batch"], cfg["heads"], cfg["seq_len"], cfg["head_dim"]
            label = f"B={B} H={H} N={N} D={D}"
            print(f"\n── {label} ─")

            # Generate inputs
            q = torch.randn(B, H, N, D, device="cuda", dtype=self.dtype)
            k = torch.randn(B, H, N, D, device="cuda", dtype=self.dtype)
            v = torch.randn(B, H, N, D, device="cuda", dtype=self.dtype)

            flops = compute_attention_flops(B, H, N, D)
            mem = compute_attention_bytes(B, H, N, D, dtype_bytes=2)
            byt = mem["flash_bytes"]  # approximate for flash variants

            size_result = SizeResult(batch=B, heads=H, seq_len=N, head_dim=D,
                                     label=label, memory=mem)

            if self.gpu_spec:
                ai = flops / byt if byt > 0 else float("inf")
                size_result.roofline = {
                    "ceiling_tflops": self.gpu_spec.peak_fp16_tflops,
                    "ceiling_bw_gbs": self.gpu_spec.hbm_bandwidth_gb_s,
                    "arithmetic_intensity": ai,
                    "bottleneck": "compute_bound" if ai >= self.gpu_spec.ridge_point_fp16 else "memory_bound",
                }

            for name, fn in impls.items():
                if fn is None:
                    print(f"  {name:<30s}  (not available)")
                    continue

                is_ref = (name == "torch SDPA (FlashAttn)")
                if not is_ref and ref_fn:
                    correct, max_diff, err = self._check(fn, ref_fn, (q, k, v), {})
                    if not correct:
                        print(f"  {name:<30s}  ❌ FAIL: max_diff={max_diff:.4e}")
                        size_result.results.append(AttentionResult(
                            name=name, time_ms=0, correct=False,
                            max_diff=max_diff, error_msg=err))
                        continue
                else:
                    correct, max_diff = True, 0.0

                try:
                    time_ms, time_std = self._time_fn(fn, (q, k, v), {})
                except torch.cuda.OutOfMemoryError:
                    print(f"  {name:<30s}  ❌ OOM (N={N} too large for naive attn)")
                    size_result.results.append(AttentionResult(
                        name=name, time_ms=0, correct=False,
                        error_msg="OOM"))
                    continue
                except Exception as e:
                    print(f"  {name:<30s}  ❌ ERROR: {e}")
                    continue

                tflops = flops / (time_ms * 1e-3) / 1e12 if time_ms > 0 else 0.0
                bw = byt / (time_ms * 1e-3) / 1e9 if time_ms > 0 else 0.0
                pct = (tflops / self.gpu_spec.peak_fp16_tflops * 100
                       if self.gpu_spec else 0.0)

                status = "✅" if correct else "❌"
                print(f"  {name:<30s}  {time_ms:>8.4f}ms ±{time_std:.4f}  "
                      f"{tflops:>6.1f} TFLOPS  {pct:>5.1f}% ceil  {status}")

                size_result.results.append(AttentionResult(
                    name=name, time_ms=time_ms, time_std_ms=time_std,
                    tflops=tflops, bandwidth_gbs=bw,
                    pct_of_ceiling=pct, max_diff=max_diff, correct=correct))

            all_results.append(size_result)

        self.print_summary(all_results)
        return all_results

    def print_summary(self, results: list[SizeResult]):
        if not results:
            return

        print(f"\n{'='*100}")
        print("  SUMMARY — Attention Performance")
        print(f"{'='*100}")

        print(f"  Memory: naive O(N²) → Flash O(N)")
        print(f"  For N=4096: naive writes {results[-1].memory.get('naive_bytes', 0)/1e9:.1f} GB "
              f"vs flash {results[-1].memory.get('flash_bytes', 0)/1e6:.1f} MB "
              f"({results[-1].memory.get('reduction_ratio', 0):.0f}x reduction)")
        print()

        # Print table
        all_names: list[str] = []
        for sr in results:
            for r in sr.results:
                if r.name not in all_names:
                    all_names.append(r.name)

        print(f"  {'Seq Len':>8s}  ", end="")
        for name in all_names:
            print(f"{name[:18]:>18s}  ", end="")
        print()
        print(f"  {'-'*8}  " + "  ".join(f"{'-'*18}" for _ in all_names))

        for sr in results:
            print(f"  {sr.seq_len:>8d}  ", end="")
            for name in all_names:
                found = [r for r in sr.results if r.name == name]
                if found and found[0].correct:
                    print(f"{found[0].time_ms:>8.4f}ms ({found[0].pct_of_ceiling:>4.0f}%)  ", end="")
                else:
                    print(f"{'N/A':>18s}  ", end="")
            print()

        if self.gpu_spec:
            print(f"\n  Roofline ceiling: {self.gpu_spec.peak_fp16_tflops:.0f} TFLOPS (fp16)")
            print(f"  Attention bottleneck: memory-bound for long sequences, "
                  f"but Flash Attention tilies to stay near compute-bound.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Attention benchmark: Flash Attention vs SDPA vs naive"
    )
    parser.add_argument("--quick", "-q", action="store_true",
                        help="Fast mode: fewer seq lengths and reps")
    parser.add_argument("--save", "-s", action="store_true",
                        help="Save results to JSON")
    parser.add_argument("--output", "-o", default="benchmarks/results",
                        help="Output directory")
    parser.add_argument("--kernel", "-k", default=None,
                        help="Filter to specific kernel (substring)")
    args = parser.parse_args()

    bench = AttentionBenchmark(
        warmup=5 if args.quick else 10,
        rep=10 if args.quick else 50,
        quick=args.quick,
    )

    results = bench.run(kernel_filter=args.kernel)

    if args.save:
        os.makedirs(args.output, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        gpu_name = bench.gpu_spec.name.replace(" ", "_") if bench.gpu_spec else "unknown"
        json_path = os.path.join(args.output, f"attention_{timestamp}_{gpu_name}.json")

        json_data = {
            "timestamp": timestamp,
            "gpu": bench.gpu_spec.name if bench.gpu_spec else "unknown",
            "benchmark": "attention",
            "dtype": "fp16",
            "results": []
        }
        for sr in results:
            entry = {
                "batch": sr.batch, "heads": sr.heads,
                "seq_len": sr.seq_len, "head_dim": sr.head_dim,
                "label": sr.label,
                "implementations": {},
                "memory": {k: v for k, v in sr.memory.items() if isinstance(v, (int, float))},
            }
            for r in sr.results:
                entry["implementations"][r.name] = {
                    "time_ms": r.time_ms, "tflops": r.tflops,
                    "pct_of_ceiling": r.pct_of_ceiling, "correct": r.correct,
                }
            json_data["results"].append(entry)

        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"\nResults saved to: {json_path}")


if __name__ == "__main__":
    main()
