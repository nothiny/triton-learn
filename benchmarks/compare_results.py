#!/usr/bin/env python3
"""
Compare two benchmark result JSON files.

Useful for tracking performance changes across code iterations.

Usage:
  python benchmarks/compare_results.py results/matmul_20250609_v1.json results/matmul_20250609_v2.json
"""

import argparse
import json
import sys
from pathlib import Path


def load_results(path: str) -> dict:
    """Load a benchmark results JSON file."""
    with open(path) as f:
        return json.load(f)


def compare_matmul(before: dict, after: dict) -> list[dict]:
    """Compare two matmul benchmark result files."""
    diffs = []
    before_results = {r["label"]: r for r in before.get("results", [])}
    after_results = {r["label"]: r for r in after.get("results", [])}

    for label in sorted(set(before_results) | set(after_results)):
        br = before_results.get(label)
        ar = after_results.get(label)
        if not br or not ar:
            continue

        for name in set(br.get("implementations", {})) & set(ar.get("implementations", {})):
            bi = br["implementations"][name]
            ai = ar["implementations"][name]

            if bi.get("time_ms", 0) > 0 and ai.get("time_ms", 0) > 0:
                speedup = bi["time_ms"] / ai["time_ms"]
                direction = "faster" if speedup > 1 else "slower"
                diffs.append({
                    "label": label,
                    "kernel": name,
                    "before_ms": bi["time_ms"],
                    "after_ms": ai["time_ms"],
                    "speedup": speedup,
                    "direction": direction,
                    "before_tflops": bi.get("tflops", 0),
                    "after_tflops": ai.get("tflops", 0),
                })

    return diffs


def compare_generic(before: dict, after: dict) -> list[dict]:
    """Generic comparison for any benchmark JSON format."""
    diffs = []
    before_results = {(r.get("label", r.get("case", ""))): r for r in before.get("results", [])}
    after_results = {(r.get("label", r.get("case", ""))): r for r in after.get("results", [])}

    for label in sorted(set(before_results) | set(after_results)):
        br = before_results.get(label)
        ar = after_results.get(label)
        if not br or not ar:
            continue

        bi = br.get("implementations", br)
        ai = ar.get("implementations", ar)

        if isinstance(bi, dict) and isinstance(ai, dict):
            for name in set(bi) & set(ai):
                bi_k = bi[name]
                ai_k = ai[name]
                if isinstance(bi_k, dict) and isinstance(ai_k, dict):
                    t_before = bi_k.get("time_ms", 0)
                    t_after = ai_k.get("time_ms", 0)
                    if t_before > 0 and t_after > 0:
                        speedup = t_before / t_after
                        diffs.append({
                            "label": label,
                            "kernel": name,
                            "before_ms": t_before,
                            "after_ms": t_after,
                            "speedup": speedup,
                            "direction": "faster" if speedup > 1 else "slower",
                        })

    return diffs


def print_diff_table(diffs: list[dict]) -> None:
    """Print a comparison table."""
    if not diffs:
        print("No comparable results found.")
        return

    print(f"\n{'='*100}")
    print(f"  Benchmark Comparison ({len(diffs)} data points)")
    print(f"{'='*100}")
    print(f"  {'Size':>15s}  {'Kernel':<30s}  {'Before':>10s}  {'After':>10s}  "
          f"{'Speedup':>8s}  {'Change':>10s}")
    print(f"  {'-'*15}  {'-'*30}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*10}")

    faster = 0
    slower = 0
    total_speedup = 0.0

    for d in sorted(diffs, key=lambda x: x["speedup"], reverse=True):
        direction_mark = "🔥" if d["speedup"] > 1.05 else ("✅" if d["speedup"] > 0.95 else "❌")
        change_pct = (d["speedup"] - 1) * 100
        print(f"  {d['label']:>15s}  {d['kernel']:<30s}  "
              f"{d['before_ms']:>8.4f}ms  {d['after_ms']:>8.4f}ms  "
              f"{d['speedup']:>6.2f}x  {direction_mark} {change_pct:>+6.1f}%")

        if d["speedup"] > 1.0:
            faster += 1
        else:
            slower += 1
        total_speedup += d["speedup"]

    print(f"\n  Summary: {faster} faster, {slower} slower")
    if diffs:
        geomean = total_speedup / len(diffs)
        print(f"  Average speedup: {geomean:.3f}x "
              f"({'improvement' if geomean > 1 else 'regression'})")


def main():
    parser = argparse.ArgumentParser(
        description="Compare two benchmark result JSON files"
    )
    parser.add_argument("before", help="Path to older benchmark results JSON")
    parser.add_argument("after", help="Path to newer benchmark results JSON")
    parser.add_argument("--benchmark", "-b", default=None,
                        choices=["matmul", "attention", "elementwise"],
                        help="Benchmark type (auto-detected if omitted)")
    args = parser.parse_args()

    before = load_results(args.before)
    after = load_results(args.after)

    benchmark_type = args.benchmark or before.get("benchmark", "generic")

    if benchmark_type == "matmul":
        diffs = compare_matmul(before, after)
    else:
        diffs = compare_generic(before, after)

    print(f"  Before: {args.before}")
    print(f"  After:  {args.after}")
    print(f"  GPU:    {before.get('gpu', 'unknown')}")
    print(f"  Type:   {benchmark_type}")

    print_diff_table(diffs)


if __name__ == "__main__":
    main()
