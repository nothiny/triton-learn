"""
Benchmark case definitions.

Each case defines:
  - What Triton kernel to test vs what PyTorch/cuBLAS reference
  - Input generators for different problem sizes
  - FLOPs and memory bandwidth calculators
  - Expected dtype and constraints

Add new cases by appending to BENCH_CASES.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch


# ---------------------------------------------------------------------------
# Helper: import from numeric-prefixed files
# ---------------------------------------------------------------------------

def _load_fn(module_path: str, func_name: str) -> Callable:
    """Load a function from a Python file with numeric prefix in filename."""
    spec = importlib.util.spec_from_file_location(
        module_path.replace("/", "_").replace(".", "_"),
        module_path + ".py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, func_name)


# ---------------------------------------------------------------------------
# Case definition
# ---------------------------------------------------------------------------


@dataclass
class BenchCase:
    """A single benchmark case: Triton kernel vs reference."""

    name: str                          # Display name
    category: str                      # "elementwise", "reduction", "gemm", "attention", "normalization"
    triton_fn: Optional[Callable]      # Triton wrapper (None if skip)
    ref_fn: Callable                   # PyTorch/cuBLAS reference
    input_gen: Callable[[int], tuple]  # (size_idx) → (args_tuple, kwargs_dict)
    flops_calc: Callable[[Any], int]   # (inputs) → total FLOPs per call
    bytes_calc: Callable[[Any], int]   # (inputs) → total bytes read+written
    sizes: list[int]                   # Problem sizes (generic index)
    size_labels: list[str] = field(default_factory=list)  # Human-readable labels
    warmup: int = 10
    rep: int = 100
    rtol: float = 1e-2
    atol: float = 1e-2

    def __post_init__(self):
        if not self.size_labels:
            self.size_labels = [str(s) for s in self.sizes]


# ---------------------------------------------------------------------------
# Build cases
# ---------------------------------------------------------------------------


def build_cases() -> list[BenchCase]:
    """Build all benchmark cases, auto-loading Triton kernels."""

    cases: list[BenchCase] = []

    # ================================================================
    # Phase 1: Elementwise & Reduction
    # ================================================================

    # -- Vector Add --
    try:
        vector_add = _load_fn("phase1_fundamentals/01_vector_add", "vector_add")
    except Exception:
        vector_add = None

    def gen_vecadd(size: int):
        x = torch.rand(size, device="cuda", dtype=torch.float32)
        y = torch.rand(size, device="cuda", dtype=torch.float32)
        return (x, y), {}

    def flops_vecadd(inputs) -> int:
        return inputs[0].numel()  # 1 FLOP per element

    def bytes_vecadd(inputs) -> int:
        n = inputs[0].numel()
        return n * 3 * 4  # x(4B) + y(4B) + out(4B)

    cases.append(BenchCase(
        name="Vector Add (f32)",
        category="elementwise",
        triton_fn=vector_add,
        ref_fn=lambda x, y: x + y,
        input_gen=gen_vecadd,
        flops_calc=flops_vecadd,
        bytes_calc=bytes_vecadd,
        sizes=[1024, 65536, 1048576, 16777216, 67108864],
        size_labels=["1K", "64K", "1M", "16M", "64M"],
        rtol=1e-5, atol=1e-5,
    ))

    # -- Fused Softmax --
    try:
        fused_softmax = _load_fn("phase1_fundamentals/17_fused_softmax", "fused_softmax")
    except Exception:
        fused_softmax = None

    def gen_softmax(size: int):
        n_rows, n_cols = 1024, size
        x = torch.randn(n_rows, n_cols, device="cuda", dtype=torch.float32)
        return (x,), {}

    def flops_softmax(inputs) -> int:
        # exp + sub + div ≈ 5 FLOPs per element
        return inputs[0].numel() * 5

    def bytes_softmax(inputs) -> int:
        n = inputs[0].numel()
        return n * 2 * 4  # read fp32 + write fp32

    cases.append(BenchCase(
        name="Fused Softmax (1024×N)",
        category="reduction",
        triton_fn=fused_softmax,
        ref_fn=lambda x: torch.softmax(x, dim=-1),
        input_gen=gen_softmax,
        flops_calc=flops_softmax,
        bytes_calc=bytes_softmax,
        sizes=[256, 1024, 4096, 16384],
        size_labels=["1024×256", "1024×1K", "1024×4K", "1024×16K"],
        rtol=1e-3, atol=1e-4,
    ))

    # -- Fused ReLU+Bias --
    try:
        fused_relu_bias = _load_fn("phase1_fundamentals/05_fused_relu_bias", "fused_relu_bias")
    except Exception:
        fused_relu_bias = None

    def gen_relu(size: int):
        x = torch.randn(size, device="cuda", dtype=torch.float32)
        bias = torch.randn(size, device="cuda", dtype=torch.float32)
        return (x, bias), {}

    def flops_relu(inputs) -> int:
        return inputs[0].numel() * 2  # add + max

    def bytes_relu(inputs) -> int:
        n = inputs[0].numel()
        return n * 3 * 4  # x + bias (read) + out (write)

    cases.append(BenchCase(
        name="Fused ReLU+Bias",
        category="elementwise",
        triton_fn=fused_relu_bias,
        ref_fn=lambda x, bias: torch.relu(x + bias),
        input_gen=gen_relu,
        flops_calc=flops_relu,
        bytes_calc=bytes_relu,
        sizes=[65536, 1048576, 16777216, 67108864],
        size_labels=["64K", "1M", "16M", "64M"],
        rtol=1e-5, atol=1e-5,
    ))

    # -- Layer Norm --
    try:
        layer_norm = _load_fn("phase1_fundamentals/21_layer_norm", "layer_norm")
    except Exception:
        layer_norm = None

    def gen_ln(size: int):
        n_rows, n_cols = size, 1024
        x = torch.randn(n_rows, n_cols, device="cuda", dtype=torch.float32)
        w = torch.randn(n_cols, device="cuda", dtype=torch.float32)
        b = torch.randn(n_cols, device="cuda", dtype=torch.float32)
        return (x, w, b), {"eps": 1e-5}

    def flops_ln(inputs) -> int:
        # mean + var + norm + affine ≈ 8 FLOPs per element
        return inputs[0].numel() * 8

    def bytes_ln(inputs) -> int:
        x = inputs[0]
        n = x.numel()
        # x(4B) + w(4B) + b(4B) + out(4B) + intermediates
        return n * 3 * 4 + x.shape[-1] * 2 * 4

    cases.append(BenchCase(
        name="Layer Norm (N×1024)",
        category="normalization",
        triton_fn=layer_norm,
        ref_fn=lambda x, w, b, eps=1e-5: torch.nn.functional.layer_norm(x, [x.shape[-1]], w, b, eps=eps),
        input_gen=gen_ln,
        flops_calc=flops_ln,
        bytes_calc=bytes_ln,
        sizes=[256, 1024, 4096, 16384],
        size_labels=["256×1K", "1K×1K", "4K×1K", "16K×1K"],
        rtol=1e-2, atol=1e-2,  # relaxed: simplified 3-pass impl
    ))

    # ================================================================
    # Phase 1 new: RMSNorm, Activations, Dropout, Residual+Norm
    # ================================================================

    # -- RMS Norm --
    try:
        rms_norm = _load_fn("phase1_fundamentals/22_rms_norm", "rms_norm")
    except Exception:
        rms_norm = None

    def gen_rms(size: int):
        n_rows, n_cols = size, 4096
        x = torch.randn(n_rows, n_cols, device="cuda", dtype=torch.float32)
        w = torch.randn(n_cols, device="cuda", dtype=torch.float32)
        return (x, w), {"eps": 1e-5}

    def ref_rms_pytorch(x, w, eps=1e-5):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        return x * rms * w

    def flops_rms(inputs) -> int:
        return inputs[0].numel() * 5  # sq + mean + rsqrt + mul

    def bytes_rms(inputs) -> int:
        x = inputs[0]
        return x.numel() * 2 * 4 + x.shape[-1] * 4  # x + w read, out write

    cases.append(BenchCase(
        name="RMS Norm (N×4096)",
        category="normalization",
        triton_fn=rms_norm,
        ref_fn=ref_rms_pytorch,
        input_gen=gen_rms,
        flops_calc=flops_rms,
        bytes_calc=bytes_rms,
        sizes=[256, 1024, 4096, 16384],
        size_labels=["256×4K", "1K×4K", "4K×4K", "16K×4K"],
        rtol=1e-3, atol=1e-3,
    ))

    # -- SiLU Activation --
    try:
        silu = _load_fn("phase1_fundamentals/07_silu", "silu")
    except Exception:
        silu = None

    def gen_elem(size: int):
        x = torch.randn(size, device="cuda", dtype=torch.float32)
        return (x,), {}

    def flops_silu(inputs) -> int:
        return inputs[0].numel() * 5  # sigmoid(~4) + mul

    def bytes_activation(inputs) -> int:
        return inputs[0].numel() * 2 * 4  # read + write

    cases.append(BenchCase(
        name="SiLU Activation",
        category="elementwise",
        triton_fn=silu,
        ref_fn=lambda x: torch.nn.functional.silu(x),
        input_gen=gen_elem,
        flops_calc=flops_silu,
        bytes_calc=bytes_activation,
        sizes=[65536, 1048576, 16777216, 67108864],
        size_labels=["64K", "1M", "16M", "64M"],
        rtol=1e-4, atol=1e-4,
    ))

    # -- GELU Activation --
    try:
        gelu = _load_fn("phase1_fundamentals/08_gelu", "gelu")
    except Exception:
        gelu = None

    def flops_gelu(inputs) -> int:
        return inputs[0].numel() * 9  # x³ + mul + add + sigmoid + mul

    cases.append(BenchCase(
        name="GELU Activation",
        category="elementwise",
        triton_fn=gelu,
        ref_fn=lambda x: torch.nn.functional.gelu(x, approximate="tanh"),
        input_gen=gen_elem,
        flops_calc=flops_gelu,
        bytes_calc=bytes_activation,
        sizes=[65536, 1048576, 16777216, 67108864],
        size_labels=["64K", "1M", "16M", "64M"],
        rtol=1e-3, atol=1e-3,  # tanh approx has ~1e-3 diff
    ))

    # -- Dropout --
    try:
        dropout_fn = _load_fn("phase1_fundamentals/09_dropout", "dropout")
    except Exception:
        dropout_fn = None

    def gen_dropout(size: int):
        x = torch.randn(size, device="cuda", dtype=torch.float32)
        return (x,), {"p": 0.5}

    def ref_dropout(x, p=0.5):
        import random
        return torch.nn.functional.dropout(x, p=p, training=True)

    def triton_dropout_wrapper(x, p=0.5):
        import random
        seed = random.randint(0, 2**31 - 1)
        return dropout_fn(x, p=p, seed=seed)[0]

    def flops_dropout(inputs) -> int:
        return inputs[0].numel() * 4  # rand + compare + mul

    def bytes_dropout(inputs) -> int:
        return inputs[0].numel() * 3 * 4  # x + out + mask

    cases.append(BenchCase(
        name="Dropout (p=0.5)",
        category="elementwise",
        triton_fn=triton_dropout_wrapper if dropout_fn else None,
        ref_fn=ref_dropout,
        input_gen=gen_dropout,
        flops_calc=flops_dropout,
        bytes_calc=bytes_dropout,
        sizes=[65536, 1048576, 16777216, 67108864],
        size_labels=["64K", "1M", "16M", "64M"],
        rtol=0.2, atol=0.2,  # statistical: mask patterns differ
    ))

    # -- Fused Residual Add + LayerNorm --
    try:
        residual_add_norm = _load_fn("phase1_fundamentals/25_residual_add_norm", "residual_add_norm")
    except Exception:
        residual_add_norm = None

    def gen_residual(size: int):
        n_rows, n_cols = size, 1024
        x = torch.randn(n_rows, n_cols, device="cuda", dtype=torch.float32)
        residual = torch.randn(n_rows, n_cols, device="cuda", dtype=torch.float32)
        w = torch.randn(n_cols, device="cuda", dtype=torch.float32)
        b = torch.randn(n_cols, device="cuda", dtype=torch.float32)
        return (x, residual, w, b), {"eps": 1e-5}

    def ref_residual(x, residual, w, b, eps=1e-5):
        return torch.nn.functional.layer_norm(
            x + residual, [x.shape[-1]], w, b, eps=eps
        )

    def flops_residual(inputs) -> int:
        return inputs[0].numel() * 9  # add + mean + var + norm + affine

    def bytes_residual(inputs) -> int:
        return inputs[0].numel() * 5 * 4  # x + r + w + b + out

    cases.append(BenchCase(
        name="Residual+LayerNorm (N×1024)",
        category="normalization",
        triton_fn=residual_add_norm,
        ref_fn=ref_residual,
        input_gen=gen_residual,
        flops_calc=flops_residual,
        bytes_calc=bytes_residual,
        sizes=[256, 1024, 4096, 16384],
        size_labels=["256×1K", "1K×1K", "4K×1K", "16K×1K"],
        rtol=1e-2, atol=1e-2,
    ))

    # -- SwiGLU --
    try:
        swiglu = _load_fn("phase1_fundamentals/10_swiglu", "swiglu")
    except Exception:
        swiglu = None

    def gen_swiglu(size: int):
        gate = torch.randn(size, 4096, device="cuda", dtype=torch.float32)
        up = torch.randn(size, 4096, device="cuda", dtype=torch.float32)
        return (gate, up), {}

    cases.append(BenchCase(
        name="SwiGLU (N×4096)",
        category="elementwise",
        triton_fn=swiglu,
        ref_fn=lambda gate, up: gate * torch.nn.functional.silu(up),
        input_gen=gen_swiglu,
        flops_calc=lambda inputs: inputs[0].numel() * 6,
        bytes_calc=lambda inputs: inputs[0].numel() * 3 * 4,
        sizes=[1024, 2048, 4096],
        size_labels=["1K×4K", "2K×4K", "4K×4K"],
        rtol=1e-4, atol=1e-4,
    ))

    # -- Cross Entropy Loss --
    try:
        cross_entropy_loss = _load_fn("phase1_fundamentals/18_cross_entropy", "cross_entropy_loss")
    except Exception:
        cross_entropy_loss = None

    def gen_ce(size: int):
        logits = torch.randn(1024, size, device="cuda", dtype=torch.float32)
        labels = torch.randint(0, size, (1024,), device="cuda")
        return (logits, labels), {}

    cases.append(BenchCase(
        name="Cross Entropy (1024×C)",
        category="reduction",
        triton_fn=cross_entropy_loss,
        ref_fn=lambda logits, labels: torch.nn.functional.cross_entropy(logits, labels),
        input_gen=gen_ce,
        flops_calc=lambda inputs: inputs[0].numel() * 13,
        bytes_calc=lambda inputs: inputs[0].numel() * 4 + inputs[1].numel() * 8,
        sizes=[1024, 8192, 32000],
        size_labels=["1024×1K", "1024×8K", "1024×32K"],
        rtol=1e-3, atol=1e-3,
    ))

    # -- Rotary Position Embedding --
    try:
        apply_rotary_emb = _load_fn("phase1_fundamentals/26_rotary_embedding", "apply_rotary_emb")
        precompute_freqs_cis = _load_fn("phase1_fundamentals/26_rotary_embedding", "precompute_freqs_cis")
    except Exception:
        apply_rotary_emb = None
        precompute_freqs_cis = None

    def gen_rope(size: int):
        head_dim = 128
        x = torch.randn(4, 32, size, head_dim, device="cuda", dtype=torch.float32)
        cos, sin = precompute_freqs_cis(head_dim, size, device="cuda")
        return (x, cos, sin), {}

    def ref_rope(x, cos, sin):
        x_rotated = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).flatten(-2)
        return x * cos + x_rotated * sin

    def triton_rope_wrapper(x, cos, sin):
        return apply_rotary_emb(x, cos, sin)

    cases.append(BenchCase(
        name="Rotary Embedding (RoPE)",
        category="elementwise",
        triton_fn=triton_rope_wrapper if apply_rotary_emb else None,
        ref_fn=ref_rope,
        input_gen=gen_rope,
        flops_calc=lambda inputs: inputs[0].numel() * 3,
        bytes_calc=lambda inputs: inputs[0].numel() * 4 * 4,
        sizes=[512, 1024, 2048],
        size_labels=["N=512", "N=1K", "N=2K"],
        rtol=1e-4, atol=1e-4,
    ))

    # -- Group Norm --
    try:
        group_norm = _load_fn("phase1_fundamentals/23_group_norm", "group_norm")
    except Exception:
        group_norm = None

    def gen_gn(size: int):
        C = 256
        x = torch.randn(size, C, device="cuda", dtype=torch.float32)
        w = torch.randn(C, device="cuda")
        b = torch.randn(C, device="cuda")
        return (x, 8, w, b), {}

    def ref_gn(x, num_groups, w, b):
        return torch.nn.functional.group_norm(x, num_groups, w, b, eps=1e-5)

    cases.append(BenchCase(
        name="Group Norm (N×256, G=8)",
        category="normalization",
        triton_fn=group_norm,
        ref_fn=ref_gn,
        input_gen=gen_gn,
        flops_calc=lambda inputs: inputs[0].numel() * 8,
        bytes_calc=lambda inputs: inputs[0].numel() * 4 * 4,
        sizes=[256, 1024, 4096],
        size_labels=["256×256", "1K×256", "4K×256"],
        rtol=1e-2, atol=1e-2,
    ))

    # ================================================================
    # Phase 2: GEMM / MatMul
    # ================================================================

    # -- Naive MatMul --
    try:
        matmul_naive = _load_fn("phase2_compute/01_matmul_naive", "matmul_naive")
    except Exception:
        matmul_naive = None

    # -- Tiled MatMul (autotuned) --
    try:
        matmul_tiled = _load_fn("phase2_compute/02_matmul_tiled", "matmul_tiled")
    except Exception:
        matmul_tiled = None

    def gen_matmul(size: int):
        M = N = K = size
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)
        return (a, b), {}

    def flops_matmul(inputs) -> int:
        a, b = inputs[0], inputs[1]
        return 2 * a.shape[0] * a.shape[1] * b.shape[1]

    def bytes_matmul(inputs) -> int:
        a, b = inputs[0], inputs[1]
        M, K = a.shape
        _, N = b.shape
        # A(fp16) + B(fp16) + C(fp16)
        return M * K * 2 + K * N * 2 + M * N * 2

    matmul_sizes = [256, 512, 1024, 2048, 4096]
    matmul_labels = ["256²", "512²", "1K²", "2K²", "4K²"]

    # Triton Naive MatMul
    cases.append(BenchCase(
        name="MatMul Naive (fp16)",
        category="gemm",
        triton_fn=matmul_naive,
        ref_fn=lambda a, b: torch.mm(a, b),  # cuBLAS
        input_gen=gen_matmul,
        flops_calc=flops_matmul,
        bytes_calc=bytes_matmul,
        sizes=matmul_sizes,
        size_labels=matmul_labels,
        rtol=0.05, atol=0.05,
    ))

    # Triton Tiled MatMul (autotuned)
    cases.append(BenchCase(
        name="MatMul Tiled (fp16, autotuned)",
        category="gemm",
        triton_fn=matmul_tiled,
        ref_fn=lambda a, b: torch.mm(a, b),  # cuBLAS
        input_gen=gen_matmul,
        flops_calc=flops_matmul,
        bytes_calc=bytes_matmul,
        sizes=matmul_sizes,
        size_labels=matmul_labels,
        rtol=0.01, atol=0.01,
    ))

    # ================================================================
    # Phase 2: Flash Attention
    # ================================================================

    try:
        flash_attention_v1 = _load_fn("phase2_compute/04_flash_attention_v1", "flash_attention_v1")
    except Exception:
        flash_attention_v1 = None

    def gen_attn(size: int):
        B, H, N, D = 2, 8, size, 64
        q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        return (q, k, v), {}

    def flops_attn(inputs) -> int:
        q = inputs[0]
        B, H, N, D = q.shape
        # Q@K^T: 2*B*H*N*N*D, P@V: 2*B*H*N*N*D, plus softmax
        return 4 * B * H * N * N * D

    def bytes_attn(inputs) -> int:
        q = inputs[0]
        B, H, N, D = q.shape
        # Q+K+V read + O write = 4 * B*H*N*D * 2bytes (fp16)
        return 4 * B * H * N * D * 2

    # PyTorch SDPA as reference (uses FlashAttention-2 internally)
    def ref_sdpa(q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)

    cases.append(BenchCase(
        name="Flash Attention v1 (fp16)",
        category="attention",
        triton_fn=flash_attention_v1,
        ref_fn=ref_sdpa,
        input_gen=gen_attn,
        flops_calc=flops_attn,
        bytes_calc=bytes_attn,
        sizes=[128, 256, 512, 1024],
        size_labels=["N=128", "N=256", "N=512", "N=1K"],
        warmup=3, rep=20,  # attention is expensive
        rtol=0.1, atol=0.1,
    ))

    # ================================================================
    # Phase 3: liger-kernel — production-grade Triton library
    # ================================================================

    try:
        # Fix: torch.distributed.tensor must be pre-imported for liger-kernel 0.8
        import torch.distributed.tensor  # noqa: F401
        from liger_kernel.transformers.functional import (
            liger_layer_norm, liger_rms_norm, liger_swiglu,
            liger_geglu, liger_softmax, liger_cross_entropy,
        )
        _has_liger = True
    except Exception:
        _has_liger = False

    # -- LayerNorm: Ours vs Liger vs PyTorch --
    def gen_ln_liger(size: int):
        n_rows, n_cols = size, 1024
        x = torch.randn(n_rows, n_cols, device="cuda", dtype=torch.float32)
        w = torch.randn(n_cols, device="cuda", dtype=torch.float32)
        b = torch.randn(n_cols, device="cuda", dtype=torch.float32)
        return (x, w, b), {"eps": 1e-5}

    # PyTorch LayerNorm ref
    def ref_ln_pytorch(x, w, b, eps=1e-5):
        return torch.nn.functional.layer_norm(x, [x.shape[-1]], w, b, eps=eps)

    # Liger LayerNorm wrapper
    def liger_ln_wrapper(x, w, b, eps=1e-5):
        return liger_layer_norm(x, w, b, eps)

    cases.append(BenchCase(
        name="LayerNorm Liger (N×1024)",
        category="normalization",
        triton_fn=liger_ln_wrapper if _has_liger else None,
        ref_fn=ref_ln_pytorch,
        input_gen=gen_ln_liger,
        flops_calc=flops_ln,
        bytes_calc=bytes_ln,
        sizes=[256, 1024, 4096, 16384],
        size_labels=["256×1K", "1K×1K", "4K×1K", "16K×1K"],
        rtol=1e-3, atol=1e-3,
    ))

    # -- RMSNorm: Liger vs PyTorch --
    def gen_rms(size: int):
        n_rows, n_cols = size, 4096
        x = torch.randn(n_rows, n_cols, device="cuda", dtype=torch.float32)
        w = torch.randn(n_cols, device="cuda", dtype=torch.float32)
        return (x, w), {"eps": 1e-5}

    def ref_rms_pytorch(x, w, eps=1e-5):
        # RMSNorm: x * rsqrt(mean(x²) + eps) * w
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        return x * rms * w

    def liger_rms_wrapper(x, w, eps=1e-5):
        return liger_rms_norm(x, w, eps)

    cases.append(BenchCase(
        name="RMSNorm Liger (N×4096)",
        category="normalization",
        triton_fn=liger_rms_wrapper if _has_liger else None,
        ref_fn=ref_rms_pytorch,
        input_gen=gen_rms,
        flops_calc=lambda inputs: inputs[0].numel() * 6,  # pow2+mean+rsqrt+mul+mul
        bytes_calc=lambda inputs: inputs[0].numel() * 3 * 4 + inputs[0].shape[-1] * 4,
        sizes=[256, 1024, 4096, 16384],
        size_labels=["256×4K", "1K×4K", "4K×4K", "16K×4K"],
        rtol=1e-4, atol=1e-4,
    ))

    # -- SwiGLU: Liger vs PyTorch --
    def gen_swiglu(size: int):
        N, D = size, 4096
        a = torch.randn(N, D, device="cuda", dtype=torch.float32)  # gate
        b = torch.randn(N, D, device="cuda", dtype=torch.float32)  # up
        return (a, b), {}

    def ref_swiglu_pytorch(a, b):
        # SwiGLU(a, b) = a * SiLU(b) where a=gate, b=up (standard convention)
        return a * torch.nn.functional.silu(b)

    def liger_swiglu_wrapper(a, b):
        # NOTE: liger convention is swiglu(up, gate) — swapped order!
        return liger_swiglu(b, a)

    cases.append(BenchCase(
        name="SwiGLU Liger (N×4096)",
        category="elementwise",
        triton_fn=liger_swiglu_wrapper if _has_liger else None,
        ref_fn=ref_swiglu_pytorch,
        input_gen=gen_swiglu,
        flops_calc=lambda inputs: inputs[0].numel() * 8,  # silu(~5) + mul
        bytes_calc=lambda inputs: inputs[0].numel() * 3 * 4,  # a+b read + out write
        sizes=[256, 1024, 4096, 16384],
        size_labels=["256×4K", "1K×4K", "4K×4K", "16K×4K"],
        rtol=1e-4, atol=1e-4,
    ))

    # -- GeGLU: Liger vs PyTorch --
    def ref_geglu_pytorch(a, b):
        return a * torch.nn.functional.gelu(b)

    def liger_geglu_wrapper(a, b):
        # NOTE: liger convention is geglu(up, gate) — swapped order!
        return liger_geglu(b, a)

    cases.append(BenchCase(
        name="GeGLU Liger (N×4096)",
        category="elementwise",
        triton_fn=liger_geglu_wrapper if _has_liger else None,
        ref_fn=ref_geglu_pytorch,
        input_gen=gen_swiglu,  # same input shape as SwiGLU
        flops_calc=lambda inputs: inputs[0].numel() * 10,  # gelu(~7) + mul
        bytes_calc=lambda inputs: inputs[0].numel() * 3 * 4,
        sizes=[256, 1024, 4096, 16384],
        size_labels=["256×4K", "1K×4K", "4K×4K", "16K×4K"],
        rtol=1e-2, atol=1e-2,  # GELU tanh approx has ~1e-3 natural diff
    ))

    # -- Softmax: Liger vs Ours vs PyTorch --
    def gen_softmax_liger(size: int):
        n_rows, n_cols = 1024, size
        x = torch.randn(n_rows, n_cols, device="cuda", dtype=torch.float32)
        return (x,), {}

    def ref_softmax_pytorch(x):
        return torch.softmax(x, dim=-1)

    def liger_softmax_wrapper(x):
        return liger_softmax(x)

    cases.append(BenchCase(
        name="Softmax Liger (1024×N)",
        category="reduction",
        triton_fn=liger_softmax_wrapper if _has_liger else None,
        ref_fn=ref_softmax_pytorch,
        input_gen=gen_softmax_liger,
        flops_calc=lambda inputs: inputs[0].numel() * 5,
        bytes_calc=lambda inputs: inputs[0].numel() * 2 * 4,
        sizes=[1024, 4096, 16384, 65536],
        size_labels=["1024×1K", "1024×4K", "1024×16K", "1024×64K"],
        rtol=1e-3, atol=1e-3,
    ))

    return cases
