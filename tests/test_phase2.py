"""Tests for Phase 2: Core Compute Kernels."""

import importlib
import sys
import pytest
import torch

sys.path.insert(0, ".")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA GPU required"
)


def _import_func(module_path: str, func_name: str):
    """Import a function from a module with a numeric-prefixed filename."""
    spec = importlib.util.spec_from_file_location(
        module_path.replace("/", "_").replace(".", "_"),
        module_path + ".py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, func_name)


class TestMatmulNaive:
    """Test 01_matmul_naive.py"""

    def test_correctness_small(self):
        matmul_naive = _import_func("phase2_compute/01_matmul_naive", "matmul_naive")
        M, N, K = 128, 128, 128
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)
        c = matmul_naive(a, b)
        expected = torch.mm(a, b)
        assert torch.allclose(c.float(), expected.float(), rtol=0.01, atol=0.01), \
            "Naive matmul mismatch"


class TestMatmulTiled:
    """Test 02_matmul_tiled.py"""

    @pytest.mark.slow
    def test_correctness_small(self):
        matmul_tiled = _import_func("phase2_compute/02_matmul_tiled", "matmul_tiled")
        M, N, K = 256, 256, 256
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)
        c = matmul_tiled(a, b)
        expected = torch.mm(a, b)
        assert torch.allclose(c.float(), expected.float(), rtol=0.01, atol=0.01), \
            "Tiled matmul mismatch"


class TestFlashAttentionV1:
    """Test 04_flash_attention_v1.py"""

    def test_correctness(self):
        flash_attention_v1 = _import_func(
            "phase2_compute/04_flash_attention_v1", "flash_attention_v1"
        )
        ref_attention = _import_func(
            "phase2_compute/04_flash_attention_v1", "ref_attention"
        )
        B, H, N, D = 1, 2, 128, 64
        q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        out = flash_attention_v1(q, k, v)
        expected = ref_attention(q.float(), k.float(), v.float()).half()
        max_diff = (out.float() - expected.float()).abs().max().item()
        assert max_diff < 0.1, f"Flash attention mismatch: max_diff={max_diff:.4f}"
