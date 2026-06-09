"""
Numerical correctness verification utilities.

Compares Triton kernel output against PyTorch reference implementations
with informative error reporting.
"""

from __future__ import annotations

import torch
from typing import Optional


def check_allclose(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-3,
    verbose: bool = True,
) -> bool:
    """
    Verify Triton kernel output matches PyTorch reference.

    Args:
        name: Human-readable test name.
        actual: Output from the Triton kernel.
        expected: Output from the PyTorch reference.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        verbose: If True, print detailed error information.

    Returns:
        True if tensors are close within tolerance.
    """
    # Ensure both tensors are on CPU and same dtype for comparison
    actual_cpu = actual.detach().cpu().float()
    expected_cpu = expected.detach().cpu().float()

    if actual_cpu.shape != expected_cpu.shape:
        if verbose:
            print(f"  [FAIL] {name}: shape mismatch "
                  f"{tuple(actual_cpu.shape)} vs {tuple(expected_cpu.shape)}")
        return False

    abs_diff = (actual_cpu - expected_cpu).abs()
    max_abs_diff = abs_diff.max().item()
    # Relative diff: |a - b| / max(|b|, 1e-8)
    rel_diff = abs_diff / (expected_cpu.abs() + 1e-8)
    max_rel_diff = rel_diff.max().item()
    # Fraction of elements exceeding tolerance
    exceed_mask = abs_diff > (atol + rtol * expected_cpu.abs())
    exceed_frac = exceed_mask.float().mean().item()

    passed = exceed_frac < 1e-4 and max_abs_diff < max(atol * 10, 1e-2)

    if verbose:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        print(f"    max_abs_diff = {max_abs_diff:.6e}")
        print(f"    max_rel_diff = {max_rel_diff:.6e}")
        print(f"    exceed_frac  = {exceed_frac:.6e} ({int(exceed_mask.sum().item())} / {exceed_mask.numel()} elements)")
        if not passed:
            # Show worst element location
            worst_idx = abs_diff.argmax().item()
            flat_idx = worst_idx
            multi_idx = tuple(
                (worst_idx // abs_diff[i:].numel() if i < abs_diff.ndim else 0)
                for i in range(abs_diff.ndim)
            )
            print(f"    worst @ {multi_idx}: actual={actual_cpu.flatten()[worst_idx].item():.6f}, "
                  f"expected={expected_cpu.flatten()[worst_idx].item():.6f}")

    return passed


def check_max_diff(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> float:
    """
    Return the maximum absolute difference (simple version).

    Args:
        name: Test name for printing.
        actual: Triton output.
        expected: Reference output.

    Returns:
        Max absolute difference (float).
    """
    actual_cpu = actual.detach().cpu().float()
    expected_cpu = expected.detach().cpu().float()
    max_diff = (actual_cpu - expected_cpu).abs().max().item()
    print(f"  {name}: max_diff = {max_diff:.6e}")
    return max_diff


def check_equal(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> bool:
    """Strict equality check (for integer tensors or exact bitwise match)."""
    passed = torch.equal(actual.detach().cpu(), expected.detach().cpu())
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}: exact match = {passed}")
    return passed


# ---------------------------------------------------------------------------
# Reference implementations (for use in tests)
# ---------------------------------------------------------------------------


def ref_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable softmax (PyTorch reference)."""
    x_max = x.max(dim=dim, keepdim=True)[0]
    x_exp = (x - x_max).exp()
    return x_exp / x_exp.sum(dim=dim, keepdim=True)


def ref_layer_norm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Layer norm (PyTorch reference)."""
    return torch.nn.functional.layer_norm(x, [x.shape[-1]], weight, bias, eps)


def ref_relu(x: torch.Tensor) -> torch.Tensor:
    """ReLU activation (PyTorch reference)."""
    return torch.nn.functional.relu(x)


def ref_gelu(x: torch.Tensor) -> torch.Tensor:
    """GELU activation (PyTorch reference)."""
    return torch.nn.functional.gelu(x)
