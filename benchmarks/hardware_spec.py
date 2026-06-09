#!/usr/bin/env python3
"""
GPU Hardware Specification Printer.

Auto-detects the current GPU and prints its complete specification:
peak TFLOPS, HBM bandwidth, L2 cache, shared memory, register file,
and roofline ridge points. Used as a prelude to any benchmark run.

Usage:
  python benchmarks/hardware_spec.py
  python benchmarks/hardware_spec.py --json  # machine-readable output
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from utils.roofline import get_gpu_spec, GPUSpec


def print_text_spec(spec: GPUSpec) -> None:
    """Print a human-readable GPU specification table."""
    print("=" * 60)
    print("  GPU Hardware Specification")
    print("=" * 60)
    print(f"  Device:              {spec.name}")
    print(f"  CUDA Capability:     {torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}")
    print(f"  SMs:                 {spec.sm_count}")
    print(f"  Peak FP16 TFLOPS:    {spec.peak_fp16_tflops:.1f}  (with sparsity: {spec.peak_fp16_tflops * 2:.1f})")
    print(f"  Peak BF16 TFLOPS:    {spec.peak_bf16_tflops:.1f}")
    print(f"  Peak FP32 TFLOPS:    {spec.peak_fp32_tflops:.1f}")
    print(f"  HBM Bandwidth:       {spec.hbm_bandwidth_gb_s:.0f} GB/s")
    if spec.l2_cache_mb > 0:
        print(f"  L2 Cache:            {spec.l2_cache_mb:.0f} MB")
    if spec.max_shared_mem_per_sm_kb > 0:
        print(f"  Shared Memory/SM:    {spec.max_shared_mem_per_sm_kb:.0f} KB (max configurable)")
    print(f"  Registers/SM:        {spec.registers_per_sm}")
    print()
    print("  --- Roofline Ridge Points ---")
    print(f"  FP16 Tensor Core:    {spec.ridge_point_fp16:.1f} FLOP/byte  (compute bound above this)")
    print(f"  FP32 CUDA Core:      {spec.ridge_point_fp32:.1f} FLOP/byte")
    print()
    print("  Note: GEMM at reasonable sizes is compute-bound.")
    print("        Elementwise ops are almost always memory-bound.")
    print("=" * 60)


def get_json_spec(spec: GPUSpec) -> dict:
    """Return GPU spec as a JSON-serializable dict."""
    return {
        "device_name": spec.name,
        "sm_count": spec.sm_count,
        "cuda_capability": f"{torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}",
        "peak_fp16_tflops": spec.peak_fp16_tflops,
        "peak_bf16_tflops": spec.peak_bf16_tflops,
        "peak_fp32_tflops": spec.peak_fp32_tflops,
        "hbm_bandwidth_gb_s": spec.hbm_bandwidth_gb_s,
        "l2_cache_mb": spec.l2_cache_mb,
        "shared_memory_per_sm_kb": spec.max_shared_mem_per_sm_kb,
        "registers_per_sm": spec.registers_per_sm,
        "ridge_point_fp16": round(spec.ridge_point_fp16, 1),
        "ridge_point_fp32": round(spec.ridge_point_fp32, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Print GPU hardware specification")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU detected.")
        sys.exit(1)

    spec = get_gpu_spec()
    if spec is None:
        print("ERROR: Could not determine GPU spec.")
        sys.exit(1)

    if args.json:
        print(json.dumps(get_json_spec(spec), indent=2))
    else:
        print_text_spec(spec)


if __name__ == "__main__":
    main()
