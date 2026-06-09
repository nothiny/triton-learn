"""
GPU hardware specification library and roofline model analysis.

Provides peak FLOPs / bandwidth for common GPUs, auto-detection, and
roofline bottleneck analysis (compute-bound vs memory-bound).

Usage::

    from utils.roofline import get_gpu_spec, roofline_analysis

    spec = get_gpu_spec()
    result = roofline_analysis(flops=2*M*N*K, bytes_accessed=bytes_total, time_ms=t)
    print(f"Bottleneck: {result['bottleneck']}")
    print(f"Efficiency: {result['efficiency_pct']:.1f}%")
"""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# GPU hardware specifications
# ---------------------------------------------------------------------------
# Values are approximate peak numbers from NVIDIA whitepapers / datasheets.
# Peak TFLOPS is for Tensor Core FP16 (dense, not sparse).
# For FP32, use the corresponding CUDA core peak.
# HBM bandwidth is the rated bidirectional bandwidth.
# NVIDIA does not expose these via CUDA API — this table is manually maintained.

@dataclass
class GPUSpec:
    """Peak hardware capabilities for a specific GPU model."""

    name: str
    peak_fp16_tflops: float     # FP16 Tensor Core (dense, no sparsity)
    peak_bf16_tflops: float     # BF16 Tensor Core
    peak_fp32_tflops: float     # FP32 CUDA cores
    hbm_bandwidth_gb_s: float   # HBM bandwidth (bidirectional)
    sm_count: int = 0
    l2_cache_mb: float = 0.0
    max_shared_mem_per_sm_kb: float = 0.0  # configurable max
    registers_per_sm: int = 65536           # Ampere/Hopper

    @property
    def ridge_point_fp16(self) -> float:
        """Roofline ridge point (FLOP/byte) for FP16 Tensor Core.
        Above this → compute-bound, below → memory-bound."""
        if self.hbm_bandwidth_gb_s == 0:
            return float("inf")
        # TFLOPS = T FLOP/s = 10^12 FLOP/s, BW = GB/s = 10^9 bytes/s
        # Ridge = (T FLOP/s) / (GB/s) = (T * 10^12) / (GB * 10^9)
        #       = (T / GB) * 1000  FLOP/byte
        return self.peak_fp16_tflops / self.hbm_bandwidth_gb_s * 1000

    @property
    def ridge_point_fp32(self) -> float:
        """Ridge point for FP32 CUDA cores."""
        if self.hbm_bandwidth_gb_s == 0:
            return float("inf")
        return self.peak_fp32_tflops / self.hbm_bandwidth_gb_s * 1000


# Known GPU specs (approximate, for roofline analysis)
# Sources: NVIDIA whitepapers, datasheets, techpowerup
GPU_SPECS: dict[str, GPUSpec] = {
    "H100 SXM5 80GB": GPUSpec(
        name="H100 SXM5 80GB",
        peak_fp16_tflops=989.4,    # without sparsity (1979 with)
        peak_bf16_tflops=989.4,
        peak_fp32_tflops=67.0,
        hbm_bandwidth_gb_s=3350.0, # HBM3
        sm_count=132,
        l2_cache_mb=50.0,
        max_shared_mem_per_sm_kb=228.0,
    ),
    "H100 PCIe 80GB": GPUSpec(
        name="H100 PCIe 80GB",
        peak_fp16_tflops=756.0,
        peak_bf16_tflops=756.0,
        peak_fp32_tflops=51.0,
        hbm_bandwidth_gb_s=2000.0,
        sm_count=114,
        l2_cache_mb=50.0,
        max_shared_mem_per_sm_kb=228.0,
    ),
    "H800 SXM5 80GB": GPUSpec(
        name="H800 SXM5 80GB",
        peak_fp16_tflops=989.4,
        peak_bf16_tflops=989.4,
        peak_fp32_tflops=67.0,
        hbm_bandwidth_gb_s=3350.0,
        sm_count=132,
        l2_cache_mb=50.0,
    ),
    "A100 SXM4 80GB": GPUSpec(
        name="A100 SXM4 80GB",
        peak_fp16_tflops=312.0,    # without sparsity (624 with)
        peak_bf16_tflops=312.0,
        peak_fp32_tflops=19.5,
        hbm_bandwidth_gb_s=2000.0,  # HBM2e (actually 2039 for 80GB)
        sm_count=108,
        l2_cache_mb=40.0,
        max_shared_mem_per_sm_kb=164.0,
    ),
    "A100 SXM4 40GB": GPUSpec(
        name="A100 SXM4 40GB",
        peak_fp16_tflops=312.0,
        peak_bf16_tflops=312.0,
        peak_fp32_tflops=19.5,
        hbm_bandwidth_gb_s=1555.0,
        sm_count=108,
        l2_cache_mb=40.0,
    ),
    "A100 PCIe 80GB": GPUSpec(
        name="A100 PCIe 80GB",
        peak_fp16_tflops=312.0,
        peak_bf16_tflops=312.0,
        peak_fp32_tflops=19.5,
        hbm_bandwidth_gb_s=1935.0,
        sm_count=108,
        l2_cache_mb=40.0,
    ),
    "A10": GPUSpec(
        name="A10",
        peak_fp16_tflops=125.0,
        peak_bf16_tflops=125.0,
        peak_fp32_tflops=31.2,
        hbm_bandwidth_gb_s=600.0,
        sm_count=72,
        l2_cache_mb=6.0,
    ),
    "RTX 4090": GPUSpec(
        name="RTX 4090",
        peak_fp16_tflops=330.0,    # Ada Lovelace
        peak_bf16_tflops=330.0,
        peak_fp32_tflops=82.6,
        hbm_bandwidth_gb_s=1008.0,
        sm_count=128,
        l2_cache_mb=72.0,
    ),
    "RTX 4080": GPUSpec(
        name="RTX 4080",
        peak_fp16_tflops=300.0,
        peak_bf16_tflops=300.0,
        peak_fp32_tflops=48.7,
        hbm_bandwidth_gb_s=716.8,
        sm_count=76,
        l2_cache_mb=64.0,
    ),
    "RTX 3090": GPUSpec(
        name="RTX 3090",
        peak_fp16_tflops=142.0,    # Ampere GA102 (non-sparse: half of 284)
        peak_bf16_tflops=142.0,
        peak_fp32_tflops=35.6,
        hbm_bandwidth_gb_s=936.0,
        sm_count=82,
        l2_cache_mb=6.0,
    ),
    "RTX 3080": GPUSpec(
        name="RTX 3080",
        peak_fp16_tflops=119.0,
        peak_bf16_tflops=119.0,
        peak_fp32_tflops=29.8,
        hbm_bandwidth_gb_s=760.0,
        sm_count=68,
        l2_cache_mb=5.0,
    ),
    "V100 SXM2 32GB": GPUSpec(
        name="V100 SXM2 32GB",
        peak_fp16_tflops=125.0,    # Volta Tensor Core
        peak_bf16_tflops=0.0,       # V100 no BF16
        peak_fp32_tflops=15.7,
        hbm_bandwidth_gb_s=900.0,
        sm_count=80,
        l2_cache_mb=6.0,
    ),
    "L40S": GPUSpec(
        name="L40S",
        peak_fp16_tflops=362.0,
        peak_bf16_tflops=362.0,
        peak_fp32_tflops=91.6,
        hbm_bandwidth_gb_s=864.0,
        sm_count=142,
        l2_cache_mb=96.0,
    ),
    "L4": GPUSpec(
        name="L4",
        peak_fp16_tflops=121.0,
        peak_bf16_tflops=121.0,
        peak_fp32_tflops=30.3,
        hbm_bandwidth_gb_s=300.0,
        sm_count=56,
        l2_cache_mb=48.0,
    ),
}


# ---------------------------------------------------------------------------
# GPU auto-detection
# ---------------------------------------------------------------------------


def get_gpu_spec() -> Optional[GPUSpec]:
    """
    Auto-detect current GPU and return its spec from the known table.

    Matches against ``torch.cuda.get_device_name()`` by substring.
    Falls back to a rough estimate if GPU is unknown.

    Returns:
        GPUSpec or None if no CUDA device is available.
    """
    if not torch.cuda.is_available():
        print("[roofline] No CUDA GPU detected — cannot determine hardware specs.")
        return None

    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)

    # Exact or substring match against known GPUs
    for key, spec in GPU_SPECS.items():
        if key in name or key.replace(" ", "") in name.replace(" ", ""):
            # Fill in runtime properties
            spec.sm_count = props.multi_processor_count
            return spec

    # Fallback: construct a rough estimate from SM count
    # Assume ~15 TFLOPS/SM for fp16, ~5 TFLOPS/SM for fp32 (Ampere+)
    # Assume ~15 GB/s per SM for bandwidth (rough)
    sm_count = props.multi_processor_count
    print(f"  [WARN] Unknown GPU '{name}' — using SM-count-based estimates.")
    print(f"  [WARN] These are rough; add your GPU to GPU_SPECS for accuracy.")

    return GPUSpec(
        name=name,
        peak_fp16_tflops=sm_count * 7.5,   # rough: ~7.5 TFLOPS/SM fp16
        peak_bf16_tflops=sm_count * 7.5,
        peak_fp32_tflops=sm_count * 0.5,   # rough: ~0.5 TFLOPS/SM fp32
        hbm_bandwidth_gb_s=sm_count * 15.0, # rough estimate
        sm_count=sm_count,
    )


# ---------------------------------------------------------------------------
# Roofline analysis
# ---------------------------------------------------------------------------


def roofline_analysis(
    flops: int,
    bytes_accessed: int,
    time_ms: float,
    dtype: str = "fp16",
    spec: Optional[GPUSpec] = None,
) -> dict:
    """
    Perform roofline analysis for a single kernel invocation.

    The roofline model classifies a kernel as compute-bound or memory-bound
    based on its arithmetic intensity (FLOPs / byte) relative to the GPU's
    ridge point (peak_TFLOPS / peak_bandwidth).

    Args:
        flops: Total floating-point operations per invocation.
               For GEMM: ``2 * M * N * K``; for elementwise: ``N * k``.
        bytes_accessed: Total bytes read from + written to HBM per invocation.
               NOT the algorithm's ideal byte count — use the actual DRAM
               traffic if measured (e.g. via ncu). Otherwise use algorithm
               count: ``(M*K + K*N + M*N) * dtype_size``.
        time_ms: Measured kernel time in milliseconds.
        dtype: "fp16", "bf16", or "fp32" — determines which peak TFLOPS to use.
        spec: GPU spec (auto-detected if None).

    Returns:
        Dict with keys::

            {
                "achieved_tflops": float,          # Measured TFLOPS
                "peak_tflops": float,              # Hardware peak TFLOPS
                "compute_utilization_pct": float,  # % of theoretical peak
                "achieved_bandwidth_gbs": float,   # Measured GB/s
                "peak_bandwidth_gbs": float,       # Hardware peak GB/s
                "bandwidth_utilization_pct": float, # % of peak bandwidth
                "arithmetic_intensity": float,      # FLOPs / byte
                "ridge_point": float,               # Break-even FLOP/byte
                "bottleneck": str,                  # "compute_bound" | "memory_bound"
                "efficiency_pct": float,            # % utilization of the limiting resource
            }
    """
    if spec is None:
        spec = get_gpu_spec()
    if spec is None:
        # Return partial results without ceiling analysis
        tflops = flops / (time_ms * 1e-3) / 1e12 if time_ms > 0 else 0.0
        bw = bytes_accessed / (time_ms * 1e-3) / 1e9 if time_ms > 0 else 0.0
        ai = flops / bytes_accessed if bytes_accessed > 0 else float("inf")
        return {
            "achieved_tflops": tflops,
            "peak_tflops": 0.0,
            "compute_utilization_pct": 0.0,
            "achieved_bandwidth_gbs": bw,
            "peak_bandwidth_gbs": 0.0,
            "bandwidth_utilization_pct": 0.0,
            "arithmetic_intensity": ai,
            "ridge_point": 0.0,
            "bottleneck": "unknown",
            "efficiency_pct": 0.0,
        }

    # Peak values based on dtype
    if dtype in ("fp16", "float16"):
        peak_tflops = spec.peak_fp16_tflops
        ridge = spec.ridge_point_fp16
    elif dtype in ("bf16", "bfloat16"):
        peak_tflops = spec.peak_bf16_tflops
        ridge = spec.ridge_point_fp16  # same as fp16 for most GPUs
    else:  # fp32
        peak_tflops = spec.peak_fp32_tflops
        ridge = spec.ridge_point_fp32

    peak_bw = spec.hbm_bandwidth_gb_s

    # Achieved metrics
    achieved_tflops = flops / (time_ms * 1e-3) / 1e12 if time_ms > 0 else 0.0
    achieved_bw = bytes_accessed / (time_ms * 1e-3) / 1e9 if time_ms > 0 else 0.0

    # Arithmetic intensity (FLOPs per byte moved to/from HBM)
    ai = flops / bytes_accessed if bytes_accessed > 0 else float("inf")

    # Bottleneck classification
    if ai >= ridge:
        bottleneck = "compute_bound"
        efficiency_pct = (achieved_tflops / peak_tflops * 100) if peak_tflops > 0 else 0.0
    else:
        bottleneck = "memory_bound"
        efficiency_pct = (achieved_bw / peak_bw * 100) if peak_bw > 0 else 0.0

    return {
        "achieved_tflops": round(achieved_tflops, 4),
        "peak_tflops": peak_tflops,
        "compute_utilization_pct": round(achieved_tflops / peak_tflops * 100, 1) if peak_tflops > 0 else 0.0,
        "achieved_bandwidth_gbs": round(achieved_bw, 2),
        "peak_bandwidth_gbs": peak_bw,
        "bandwidth_utilization_pct": round(achieved_bw / peak_bw * 100, 1) if peak_bw > 0 else 0.0,
        "arithmetic_intensity": round(ai, 2),
        "ridge_point": round(ridge, 2),
        "bottleneck": bottleneck,
        "efficiency_pct": round(efficiency_pct, 1),
    }


def print_roofline_report(result: dict) -> None:
    """Pretty-print a roofline analysis result."""
    print(f"  Achieved TFLOPS:     {result['achieved_tflops']:.2f}")
    print(f"  Peak TFLOPS:         {result['peak_tflops']:.1f}  "
          f"({result['compute_utilization_pct']:.1f}%)")
    print(f"  Achieved BW:         {result['achieved_bandwidth_gbs']:.1f} GB/s")
    print(f"  Peak BW:             {result['peak_bandwidth_gbs']:.1f} GB/s  "
          f"({result['bandwidth_utilization_pct']:.1f}%)")
    print(f"  Arithmetic Intensity: {result['arithmetic_intensity']:.1f} FLOP/byte")
    print(f"  Ridge Point:         {result['ridge_point']:.1f} FLOP/byte")
    print(f"  Bottleneck:          {result['bottleneck']}")
    print(f"  Efficiency:          {result['efficiency_pct']:.1f}% of limiting resource")


# ---------------------------------------------------------------------------
# Convenience: print full GPU spec
# ---------------------------------------------------------------------------


def print_gpu_spec(spec: Optional[GPUSpec] = None) -> None:
    """Print a formatted GPU specification table."""
    if spec is None:
        spec = get_gpu_spec()
    if spec is None:
        print("No GPU spec available.")
        return

    print("=" * 60)
    print(f"  GPU Hardware Specification")
    print("=" * 60)
    print(f"  Device:              {spec.name}")
    print(f"  SMs:                 {spec.sm_count}")
    print(f"  Peak FP16 TFLOPS:    {spec.peak_fp16_tflops:.1f}")
    print(f"  Peak BF16 TFLOPS:    {spec.peak_bf16_tflops:.1f}")
    print(f"  Peak FP32 TFLOPS:    {spec.peak_fp32_tflops:.1f}")
    print(f"  HBM Bandwidth:       {spec.hbm_bandwidth_gb_s:.0f} GB/s")
    if spec.l2_cache_mb > 0:
        print(f"  L2 Cache:            {spec.l2_cache_mb:.0f} MB")
    if spec.max_shared_mem_per_sm_kb > 0:
        print(f"  Shared Memory/SM:    {spec.max_shared_mem_per_sm_kb:.0f} KB (max)")
    print(f"  Registers/SM:        {spec.registers_per_sm}")
    print()
    print(f"  --- Roofline Ridge Points ---")
    print(f"  FP16 Tensor Core:    {spec.ridge_point_fp16:.1f} FLOP/byte  "
          f"(compute-bound above this)")
    print(f"  FP32 CUDA Core:      {spec.ridge_point_fp32:.1f} FLOP/byte")
    print()
    print(f"  Note: Most GEMM kernels are compute-bound at typical sizes.")
    print(f"        Elementwise ops are almost always memory-bound.")
    print("=" * 60)
