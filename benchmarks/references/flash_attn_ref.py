"""
Flash Attention reference wrapper.

Flash Attention (Dao et al., 2022/2023) is the SotA for attention kernels.
We support multiple backends in order of preference:

1. flash-attn 2.x library (Tri Dao's official implementation)
2. torch.nn.functional.scaled_dot_product_attention (PyTorch SDPA, uses
   FlashAttention-2 backend under the hood on Ampere+)
3. Standard attention (fallback, O(N²) memory)

Why flash-attn is fast:
  - Hand-written CUDA kernels for each GPU architecture (A100, H100)
  - Hopper version uses wgmma (async Tensor Core) + TMA (Tensor Memory
    Accelerator) — instructions Triton cannot currently express
  - This is why your Triton implementation will be within 70-85% of it
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F


def get_flash_attn() -> Optional[Callable]:
    """
    Try to import flash-attn library (Tri Dao's implementation).

    Returns:
        Callable ``fn(q, k, v) -> o`` with signature:
          q, k, v: (batch, n_heads, seq_len, head_dim) fp16/bf16
          returns: (batch, n_heads, seq_len, head_dim)
        None if flash-attn is not installed.
    """
    try:
        from flash_attn import flash_attn_func
    except ImportError:
        return None
    except Exception:
        return None

    def wrapper(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """flash-attn v2 forward pass."""
        # flash_attn_func expects (batch, seq_len, n_heads, head_dim)
        # Our convention is (batch, n_heads, seq_len, head_dim)
        q_t = q.transpose(1, 2).contiguous()
        k_t = k.transpose(1, 2).contiguous()
        v_t = v.transpose(1, 2).contiguous()
        out = flash_attn_func(q_t, k_t, v_t, causal=False)
        return out.transpose(1, 2).contiguous()

    return wrapper


def get_torch_sdpa() -> Callable:
    """
    PyTorch scaled_dot_product_attention — always available (PyTorch >= 2.0).

    On Ampere+ GPUs with fp16/bf16 inputs, this automatically dispatches to
    FlashAttention-2 backend.

    Returns:
        Callable ``fn(q, k, v) -> o``.
    """
    def sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """PyTorch SDPA (FlashAttention-2 or MemEfficient backend)."""
        return F.scaled_dot_product_attention(q, k, v)

    return sdpa


def get_naive_attention() -> Callable:
    """
    Standard scaled dot-product attention (O(N²) memory).
    This is the baseline — what Flash Attention improves upon.

    Returns:
        Callable ``fn(q, k, v) -> o``.
    """
    def naive(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        d_head = q.shape[-1]
        scale = 1.0 / (d_head ** 0.5)
        attn_weights = (q @ k.transpose(-2, -1)) * scale
        attn_probs = torch.softmax(attn_weights, dim=-1)
        return attn_probs @ v

    return naive


# ---------------------------------------------------------------------------
# Utility: compute theoretical memory traffic for attention
# ---------------------------------------------------------------------------


def compute_attention_flops(batch: int, heads: int, seq_len: int, head_dim: int) -> int:
    """
    Compute theoretical FLOPs for standard attention.

    - Q @ K^T: 2 * B * H * N * N * D  (multiply-add)
    - Softmax: ~5 FLOPs per element (exp + sub + div + sum + div) ≈ 5 * B * H * N * N
    - P @ V:   2 * B * H * N * N * D  (multiply-add)

    Total ≈ 4 * B * H * N² * D + 5 * B * H * N²
           ≈ 4 * B * H * N² * D  (dominated by matmuls for D ≫ 1)
    """
    matmul_flops = 4 * batch * heads * seq_len * seq_len * head_dim  # QK^T + PV
    softmax_flops = 5 * batch * heads * seq_len * seq_len
    return matmul_flops + softmax_flops


def compute_attention_bytes(
    batch: int, heads: int, seq_len: int, head_dim: int,
    dtype_bytes: int = 2,  # fp16
) -> dict:
    """
    Compute HBM traffic for naive vs flash attention.

    Returns dict with:
      - naive_bytes: O(N²) — attention matrix read + written to HBM
      - flash_bytes: O(N)  — only Q/K/V/O touched once each
    """
    # Naive: Q, K, V read + S (N²) written + S read + O written
    # = (3*N*D + N² + N² + N*D) * bytes = (4*N*D + 2*N²) * bytes
    elem_size = dtype_bytes
    naive_bytes = (4 * seq_len * head_dim + 2 * seq_len * seq_len) * elem_size * batch * heads

    # Flash: Q, K, V read + O written = 4 * N * D * bytes
    flash_bytes = 4 * seq_len * head_dim * elem_size * batch * heads

    return {
        "naive_bytes": naive_bytes,
        "flash_bytes": flash_bytes,
        "reduction_ratio": naive_bytes / flash_bytes if flash_bytes > 0 else float("inf"),
    }
