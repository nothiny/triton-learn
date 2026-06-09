"""Pytest configuration and shared fixtures."""

import pytest
import torch


def has_gpu() -> bool:
    """Check if CUDA GPU is available."""
    return torch.cuda.is_available()


def has_triton() -> bool:
    """Check if Triton is importable."""
    try:
        import triton
        return True
    except ImportError:
        return False


# Markers
pytest.mark.gpu = pytest.mark.skipif(not has_gpu(), reason="CUDA GPU required")
pytest.mark.triton = pytest.mark.skipif(not has_triton(), reason="Triton required")


@pytest.fixture(scope="session")
def device() -> str:
    """Test device: 'cuda' if available, else 'cpu'."""
    return "cuda" if has_gpu() else "cpu"


@pytest.fixture(scope="session")
def gpu_name() -> str:
    """GPU name for printing in test reports."""
    if has_gpu():
        return torch.cuda.get_device_name(0)
    return "N/A"
