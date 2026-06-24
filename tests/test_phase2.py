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
    """Test 07_flash_attention_v1.py"""

    def test_correctness(self):
        flash_attention_v1 = _import_func(
            "phase2_compute/07_flash_attention_v1", "flash_attention_v1"
        )
        ref_attention = _import_func(
            "phase2_compute/07_flash_attention_v1", "ref_attention"
        )
        B, H, N, D = 1, 2, 128, 64
        q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        out = flash_attention_v1(q, k, v)
        expected = ref_attention(q.float(), k.float(), v.float()).half()
        max_diff = (out.float() - expected.float()).abs().max().item()
        assert max_diff < 0.1, f"Flash attention mismatch: max_diff={max_diff:.4f}"


class TestFlashAttentionBackward:
    """Test 10_flash_attention_backward.py"""

    def test_forward_with_lse(self):
        fwd = _import_func(
            "phase2_compute/10_flash_attention_backward", "flash_attention_fwd_with_lse"
        )
        ref = _import_func(
            "phase2_compute/10_flash_attention_backward", "ref_attention_with_grad"
        )
        B, H, N, D = 1, 2, 128, 64
        q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        o, lse = fwd(q, k, v)
        expected = ref(q.float(), k.float(), v.float()).half()
        max_diff = (o.float() - expected.float()).abs().max().item()
        assert max_diff < 0.05, f"Forward mismatch: {max_diff:.4e}"
        assert lse.shape == (B * H, N)

    def test_backward_non_causal(self):
        bwd = _import_func(
            "phase2_compute/10_flash_attention_backward", "flash_attention_bwd"
        )
        fwd = _import_func(
            "phase2_compute/10_flash_attention_backward", "flash_attention_fwd_with_lse"
        )
        B, H, N, D = 1, 2, 128, 64
        q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        o, lse = fwd(q, k, v)
        do = torch.randn_like(o)
        dq, dk, dv = bwd(q, k, v, o, do, lse, causal=False)
        assert dq.shape == q.shape
        assert dk.shape == k.shape
        assert dv.shape == v.shape

    def test_autograd_function(self):
        flash_fn = _import_func(
            "phase2_compute/10_flash_attention_backward", "flash_attention_with_grad"
        )
        B, H, N, D = 1, 2, 128, 64
        q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16, requires_grad=True)
        k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16, requires_grad=True)
        v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16, requires_grad=True)
        o = flash_fn(q, k, v, causal=False)
        o.sum().backward()
        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None


class TestGroupedQueryAttention:
    """Test 11_grouped_query_attention.py"""

    def test_correctness_gqa(self):
        gqa = _import_func(
            "phase2_compute/11_grouped_query_attention", "grouped_query_attention"
        )
        B, Hq, Hkv, N, D = 1, 8, 2, 128, 64
        q = torch.randn(B, Hq, N, D, device="cuda", dtype=torch.float16)
        kv = torch.randn(B, Hkv, N, D, device="cuda", dtype=torch.float16)
        out = gqa(q, kv, kv, causal=False)
        expected = torch.nn.functional.scaled_dot_product_attention(
            q, kv, kv, enable_gqa=True)
        max_diff = (out.float() - expected.float()).abs().max().item()
        assert max_diff < 0.05, f"GQA mismatch: {max_diff:.4e}"

    def test_correctness_mqa(self):
        gqa = _import_func(
            "phase2_compute/11_grouped_query_attention", "grouped_query_attention"
        )
        B, Hq, Hkv, N, D = 1, 8, 1, 128, 64  # MQA: single KV head
        q = torch.randn(B, Hq, N, D, device="cuda", dtype=torch.float16)
        kv = torch.randn(B, Hkv, N, D, device="cuda", dtype=torch.float16)
        out = gqa(q, kv, kv, causal=True)
        expected = torch.nn.functional.scaled_dot_product_attention(
            q, kv, kv, enable_gqa=True, is_causal=True)
        max_diff = (out.float() - expected.float()).abs().max().item()
        assert max_diff < 0.05, f"MQA mismatch: {max_diff:.4e}"


class TestSlidingWindowAttention:
    """Test 12_sliding_window_attention.py"""

    def test_correctness_causal(self):
        swa = _import_func(
            "phase2_compute/12_sliding_window_attention", "sliding_window_attention"
        )
        ref = _import_func(
            "phase2_compute/12_sliding_window_attention", "ref_sliding_window"
        )
        B, H, N, D, win = 1, 2, 128, 64, 32
        q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        out = swa(q, k, v, window_size=win, causal=True)
        expected = ref(q.float(), k.float(), v.float(), window_size=win, causal=True).half()
        max_diff = (out.float() - expected.float()).abs().max().item()
        assert max_diff < 0.05, f"Sliding window mismatch: {max_diff:.4e}"

    def test_correctness_bidirectional(self):
        swa = _import_func(
            "phase2_compute/12_sliding_window_attention", "sliding_window_attention"
        )
        ref = _import_func(
            "phase2_compute/12_sliding_window_attention", "ref_sliding_window"
        )
        B, H, N, D, win = 1, 2, 128, 64, 32
        q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        out = swa(q, k, v, window_size=win, causal=False)
        expected = ref(q.float(), k.float(), v.float(), window_size=win, causal=False).half()
        max_diff = (out.float() - expected.float()).abs().max().item()
        assert max_diff < 0.05, f"Bidirectional SWA mismatch: {max_diff:.4e}"

    def test_full_window(self):
        """window_size >= N should degrade to full attention"""
        swa = _import_func(
            "phase2_compute/12_sliding_window_attention", "sliding_window_attention"
        )
        ref = _import_func(
            "phase2_compute/12_sliding_window_attention", "ref_sliding_window"
        )
        B, H, N, D, win = 1, 2, 128, 64, 256  # win > N
        q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        out = swa(q, k, v, window_size=win, causal=True)
        expected = ref(q.float(), k.float(), v.float(), window_size=win, causal=True).half()
        max_diff = (out.float() - expected.float()).abs().max().item()
        assert max_diff < 0.05, f"Full window mismatch: {max_diff:.4e}"


class TestAttentionBias:
    """Test 13_attention_bias.py"""

    def test_vector_bias(self):
        fab = _import_func(
            "phase2_compute/13_attention_bias", "flash_attention_with_bias"
        )
        build = _import_func(
            "phase2_compute/13_attention_bias", "build_alibi_bias"
        )
        ref = _import_func(
            "phase2_compute/13_attention_bias", "ref_attention_with_bias"
        )
        B, H, N, D = 1, 4, 128, 64
        q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        bias = build(N, H, q.device).expand(B, H, 1, N)
        out = fab(q, k, v, bias, causal=True)
        expected = ref(q.float(), k.float(), v.float(), bias.float(), causal=True).half()
        max_diff = (out.float() - expected.float()).abs().max().item()
        assert max_diff < 0.05, f"Vector bias mismatch: {max_diff:.4e}"

    def test_no_bias(self):
        fab = _import_func(
            "phase2_compute/13_attention_bias", "flash_attention_with_bias"
        )
        B, H, N, D = 1, 4, 128, 64
        q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        out = fab(q, k, v, bias=None, causal=False)
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        max_diff = (out.float() - expected.float()).abs().max().item()
        assert max_diff < 0.05, f"No-bias mismatch: {max_diff:.4e}"
