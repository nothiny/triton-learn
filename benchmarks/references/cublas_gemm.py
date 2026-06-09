"""
cuBLAS GEMM reference wrapper.

cuBLAS is the gold standard for GEMM performance on NVIDIA GPUs.
We access it through ``torch.matmul`` / ``torch.mm``, which dispatches
to cuBLAS for supported dtypes and shapes.

Note: cuBLAS is closed-source. We can't inspect its implementation,
but we can measure its performance and use ncu to see which PTX/SASS
instructions it uses (mma.sync, ld.shared, cp.async, etc.).
"""

from __future__ import annotations

from typing import Callable, Optional

import torch


def get_cublas_gemm(dtype: torch.dtype = torch.float16) -> Callable:
    """
    Return a cuBLAS GEMM reference.

    For fp16/bf16 matrices, torch.mm dispatches to cuBLAS GEMM kernels.
    For fp32, it may use cuBLAS or a custom CUDA kernel.

    Args:
        dtype: Desired compute dtype. cuBLAS will use Tensor Cores
               for fp16/bf16 on Ampere+.

    Returns:
        A callable ``fn(a, b) -> c`` where a and b are 2D tensors.
    """
    def cublas_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """cuBLAS GEMM via torch.mm."""
        return torch.mm(a.to(dtype), b.to(dtype))

    return cublas_matmul


def get_cublas_matmul_no_tf32() -> Callable:
    """
    Return cuBLAS GEMM with TF32 disabled.

    On Ampere+, cuBLAS uses TF32 for fp32 matmul by default,
    which gives higher throughput at slightly reduced precision.
    This wrapper disables TF32 for a strict fp32 baseline.

    Returns:
        A callable ``fn(a, b) -> c``.
    """
    def matmul_no_tf32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Disable TF32 override for this call
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        ):
            pass
        # Actually, torch.backends.cuda.matmul.allow_tf32 controls TF32
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            result = torch.mm(a.float(), b.float())
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32
        return result

    return matmul_no_tf32


def get_torch_matmul(dtype: torch.dtype = torch.float16) -> Optional[Callable]:
    """
    PyTorch matmul (cuBLAS) — always available.
    Included for interface consistency with other reference modules.
    """
    def ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.mm(a.to(dtype), b.to(dtype))

    return ref
