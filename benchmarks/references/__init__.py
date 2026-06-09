"""
Reference wrappers for SotA GPU kernel implementations.

Each module provides a get_*() function that returns a callable or None
if the dependency isn't available. This allows benchmarks to gracefully
degrade when optional SotA references aren't installed.
"""

from benchmarks.references.cublas_gemm import get_cublas_gemm
from benchmarks.references.flash_attn_ref import get_flash_attn, get_torch_sdpa
from benchmarks.references.liger_ref import (
    get_liger_ln, get_liger_rms_norm,
    get_liger_swiglu, get_liger_geglu, get_liger_softmax,
)

__all__ = [
    "get_cublas_gemm",
    "get_flash_attn", "get_torch_sdpa",
    "get_liger_ln", "get_liger_rms_norm",
    "get_liger_swiglu", "get_liger_geglu", "get_liger_softmax",
]
