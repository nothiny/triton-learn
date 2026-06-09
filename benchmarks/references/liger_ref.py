"""
Liger Kernel reference wrapper.

Liger Kernel (LinkedIn, 2024) is a production-grade Triton kernel library
for LLM training. It provides highly optimized implementations of:
  - LayerNorm, RMSNorm
  - SwiGLU, GeGLU (gated activation functions)
  - Fused softmax
  - Cross entropy loss
  - Fused linear + activation

These serve as "best Triton implementations" references — they show what's
achievable with pure Triton (no hand-written CUDA).

Install: pip install liger-kernel
GitHub: https://github.com/linkedin/Liger-Kernel

Note: liger-kernel's function signatures may differ from the standard
PyTorch API. Each wrapper handles the conversion.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch


def _has_liger() -> bool:
    """Check if liger-kernel is installed."""
    try:
        import torch.distributed.tensor  # noqa: F401 — pre-req for liger
        import liger_kernel  # noqa: F401
        return True
    except ImportError:
        return False


def get_liger_ln() -> Optional[Callable]:
    """
    Liger LayerNorm.

    Returns:
        ``fn(x, w, b, eps=1e-5) -> out`` or None.
    """
    if not _has_liger():
        return None

    from liger_kernel.transformers.functional import liger_layer_norm

    def wrapper(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor,
                eps: float = 1e-5) -> torch.Tensor:
        return liger_layer_norm(x, w, b, eps)

    return wrapper


def get_liger_rms_norm() -> Optional[Callable]:
    """
    Liger RMSNorm.

    Returns:
        ``fn(x, w, eps=1e-5) -> out`` or None.
    """
    if not _has_liger():
        return None

    from liger_kernel.transformers.functional import liger_rms_norm

    def wrapper(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        return liger_rms_norm(x, w, eps)

    return wrapper


def get_liger_swiglu() -> Optional[Callable]:
    """
    Liger SwiGLU.

    NOTE: liger's convention is swiglu(up, gate), NOT swiglu(gate, up).
    This wrapper handles the parameter swap.

    Returns:
        ``fn(gate, up) -> out`` where out = gate * SiLU(up), or None.
    """
    if not _has_liger():
        return None

    from liger_kernel.transformers.functional import liger_swiglu

    def wrapper(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        # liger convention: swiglu(up, gate) — swapped!
        return liger_swiglu(up, gate)

    return wrapper


def get_liger_geglu() -> Optional[Callable]:
    """
    Liger GeGLU.

    NOTE: same parameter swap as SwiGLU.

    Returns:
        ``fn(gate, up) -> out`` where out = gate * GELU(up), or None.
    """
    if not _has_liger():
        return None

    from liger_kernel.transformers.functional import liger_geglu

    def wrapper(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return liger_geglu(up, gate)

    return wrapper


def get_liger_softmax() -> Optional[Callable]:
    """
    Liger fused softmax.

    Returns:
        ``fn(x) -> out`` or None.
    """
    if not _has_liger():
        return None

    from liger_kernel.transformers.functional import liger_softmax

    def wrapper(x: torch.Tensor) -> torch.Tensor:
        return liger_softmax(x)

    return wrapper


def get_liger_cross_entropy() -> Optional[Callable]:
    """
    Liger fused cross entropy loss.

    Returns:
        ``fn(logits, labels) -> loss`` or None.
    """
    if not _has_liger():
        return None

    from liger_kernel.transformers.functional import liger_cross_entropy

    def wrapper(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return liger_cross_entropy(logits, labels)

    return wrapper
