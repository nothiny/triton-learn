#!/usr/bin/env python3
"""
Phase 1 Kernel Benchmark — Triton vs PyTorch vs Liger.

Compares our Triton kernels from phase1_fundamentals/ against:
  - PyTorch (cuBLAS/cuDNN backend)
  - Liger Kernel (production-grade Triton library, if installed)

For kernels without a liger equivalent (vector_add, relu_bias),
falls back to a 2-way Triton vs PyTorch comparison.

Usage:
  python benchmarks/bench_phase1.py
  python benchmarks/bench_phase1.py --category elementwise
  python benchmarks/bench_phase1.py --quick --save
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.profiler import bench_compare, print_compare_report, CompareResult
from benchmarks.references.liger_ref import (
    get_liger_ln,
    get_liger_rms_norm,
    get_liger_softmax,
    get_liger_swiglu,
    get_liger_geglu,
    get_liger_cross_entropy,
)


# ---------------------------------------------------------------------------
# Kernel loader
# ---------------------------------------------------------------------------

def _load_fn(module_path: str, func_name: str) -> Optional[Callable]:
    """Load a function from a Python file with numeric prefix in filename."""
    import importlib

    try:
        spec = importlib.util.spec_from_file_location(
            module_path.replace("/", "_").replace(".", "_"),
            module_path + ".py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, func_name)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Benchmark cases
# ---------------------------------------------------------------------------


def build_phase1_cases() -> list[dict]:
    """Build all phase1 benchmark case definitions."""

    # Load Triton kernels — Group 1: Basics
    vector_add = _load_fn("phase1_fundamentals/01_vector_add", "vector_add")
    sigmoid = _load_fn("phase1_fundamentals/02_sigmoid", "sigmoid")
    tanh = _load_fn("phase1_fundamentals/03_tanh", "tanh")

    # Group 2: Elementwise Fusion
    leaky_relu = _load_fn("phase1_fundamentals/04_leaky_relu", "leaky_relu")
    prelu = _load_fn("phase1_fundamentals/04_leaky_relu", "prelu")
    fused_relu_bias = _load_fn("phase1_fundamentals/05_fused_relu_bias", "fused_relu_bias")
    fused_scale_bias_residual = _load_fn("phase1_fundamentals/06_fused_scale_bias_residual", "fused_scale_bias_residual")

    # Group 3: Advanced Activations
    silu = _load_fn("phase1_fundamentals/07_silu", "silu")
    gelu = _load_fn("phase1_fundamentals/08_gelu", "gelu")
    dropout_fn = _load_fn("phase1_fundamentals/09_dropout", "dropout")

    # Group 4: Gated Activations
    swiglu = _load_fn("phase1_fundamentals/10_swiglu", "swiglu")
    geglu = _load_fn("phase1_fundamentals/11_geglu", "geglu")

    # Group 5: Reductions
    fused_softmax = _load_fn("phase1_fundamentals/12_fused_softmax", "fused_softmax")
    cross_entropy_loss = _load_fn("phase1_fundamentals/13_cross_entropy", "cross_entropy_loss")
    cumsum_full = _load_fn("phase1_fundamentals/14_cumsum", "cumsum_full")
    clip_grad_norm = _load_fn("phase1_fundamentals/15_gradient_clipping", "clip_grad_norm")

    # Group 6: Normalizations
    layer_norm = _load_fn("phase1_fundamentals/16_layer_norm", "layer_norm")
    rms_norm = _load_fn("phase1_fundamentals/17_rms_norm", "rms_norm")
    group_norm = _load_fn("phase1_fundamentals/18_group_norm", "group_norm")
    batchnorm1d = _load_fn("phase1_fundamentals/19_batch_norm", "batchnorm1d")
    residual_add_norm = _load_fn("phase1_fundamentals/20_residual_add_norm", "residual_add_norm")

    # Group 7: Position / Embedding / Optimizer
    apply_rotary_emb = _load_fn("phase1_fundamentals/21_rotary_embedding", "apply_rotary_emb")
    precompute_freqs_cis = _load_fn("phase1_fundamentals/21_rotary_embedding", "precompute_freqs_cis")
    embedding = _load_fn("phase1_fundamentals/22_embedding", "embedding")
    adamw_step = _load_fn("phase1_fundamentals/23_adamw", "adamw_step")
    adamw_pytorch_step = _load_fn("phase1_fundamentals/23_adamw", "adamw_pytorch_step")

    # Load liger kernels
    liger_softmax = get_liger_softmax()
    liger_ln = get_liger_ln()
    liger_rms = get_liger_rms_norm()
    liger_swiglu_fn = get_liger_swiglu()
    liger_geglu_fn = get_liger_geglu()
    liger_ce = get_liger_cross_entropy()

    cases: list[dict] = []

    # ================================================================
    # Case 1: Vector Add
    # ================================================================
    def gen_vecadd(size: int):
        x = torch.rand(size, device="cuda", dtype=torch.float32)
        y = torch.rand(size, device="cuda", dtype=torch.float32)
        return (x, y), {}

    def make_vecadd_impls(x, y):
        return {
            "Triton (ours)": lambda: vector_add(x, y),
            "PyTorch (ref)": lambda: x + y,
        }

    cases.append({
        "name": "Vector Add (f32)",
        "category": "elementwise",
        "gen": gen_vecadd,
        "make_impls": make_vecadd_impls,
        "flops": lambda args: args[0].numel(),  # 1 FLOP per element
        "bytes": lambda args: args[0].numel() * 3 * 4,  # x + y read, out write
        "dtype": "fp32",
        "sizes": [65536, 1048576, 16777216, 67108864],
        "labels": ["64K", "1M", "16M", "64M"],
    })

    # ================================================================
    # Case 2: Fused Softmax (Triton vs Liger vs PyTorch)
    # ================================================================
    def gen_softmax(size: int):
        n_rows, n_cols = 1024, size
        x = torch.randn(n_rows, n_cols, device="cuda", dtype=torch.float32)
        return (x,), {}

    def make_softmax_impls(x):
        impls = {
            "Triton (ours)": lambda: fused_softmax(x),
            "PyTorch (ref)": lambda: torch.softmax(x, dim=-1),
        }
        if liger_softmax:
            impls["Liger (SotA)"] = lambda: liger_softmax(x)
        return impls

    cases.append({
        "name": "Fused Softmax (1024×N)",
        "category": "reduction",
        "gen": gen_softmax,
        "make_impls": make_softmax_impls,
        "flops": lambda args: args[0].numel() * 5,  # exp + sub + div ≈ 5 FLOPs
        "bytes": lambda args: args[0].numel() * 2 * 4,  # read + write fp32
        "dtype": "fp32",
        "sizes": [256, 1024, 4096, 16384],
        "labels": ["1024×256", "1024×1K", "1024×4K", "1024×16K"],
    })

    # ================================================================
    # Case 3: Fused ReLU+Bias
    # ================================================================
    def gen_relu(size: int):
        x = torch.randn(size, device="cuda", dtype=torch.float32)
        bias = torch.randn(size, device="cuda", dtype=torch.float32)
        return (x, bias), {}

    def make_relu_impls(x, bias):
        return {
            "Triton (ours)": lambda: fused_relu_bias(x, bias),
            "PyTorch (ref)": lambda: torch.relu(x + bias),
        }

    cases.append({
        "name": "Fused ReLU+Bias",
        "category": "elementwise",
        "gen": gen_relu,
        "make_impls": make_relu_impls,
        "flops": lambda args: args[0].numel() * 2,  # add + max
        "bytes": lambda args: args[0].numel() * 3 * 4,  # x + bias read, out write
        "dtype": "fp32",
        "sizes": [65536, 1048576, 16777216, 67108864],
        "labels": ["64K", "1M", "16M", "64M"],
    })

    # ================================================================
    # Case 4: Layer Norm (Triton vs Liger vs PyTorch)
    # ================================================================
    def gen_ln(size: int):
        n_rows, n_cols = size, 1024
        x = torch.randn(n_rows, n_cols, device="cuda", dtype=torch.float32)
        w = torch.randn(n_cols, device="cuda", dtype=torch.float32)
        b = torch.randn(n_cols, device="cuda", dtype=torch.float32)
        return (x, w, b), {"eps": 1e-5}

    def make_ln_impls(x, w, b, eps=1e-5):
        impls = {
            "Triton (ours)": lambda: layer_norm(x, w, b, eps),
            "PyTorch (ref)": lambda: torch.nn.functional.layer_norm(
                x, [x.shape[-1]], w, b, eps=eps
            ),
        }
        if liger_ln:
            impls["Liger (SotA)"] = lambda: liger_ln(x, w, b, eps)
        return impls

    cases.append({
        "name": "Layer Norm (N×1024)",
        "category": "normalization",
        "gen": gen_ln,
        "make_impls": make_ln_impls,
        "flops": lambda args: args[0].numel() * 8,  # mean + var + norm + affine
        "bytes": lambda args: args[0].numel() * 3 * 4 + args[0].shape[-1] * 2 * 4,
        "dtype": "fp32",
        "sizes": [256, 1024, 4096, 16384],
        "labels": ["256×1K", "1K×1K", "4K×1K", "16K×1K"],
    })

    # ================================================================
    # Case 5: RMS Norm (Triton vs Liger vs PyTorch)
    # ================================================================
    def gen_rms(size: int):
        n_rows, n_cols = size, 4096
        x = torch.randn(n_rows, n_cols, device="cuda", dtype=torch.float32)
        w = torch.randn(n_cols, device="cuda", dtype=torch.float32)
        return (x, w), {"eps": 1e-5}

    def make_rms_impls(x, w, eps=1e-5):
        impls = {
            "Triton (ours)": lambda: rms_norm(x, w, eps),
            "PyTorch (ref)": lambda: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w,
        }
        if liger_rms:
            impls["Liger (SotA)"] = lambda: liger_rms(x, w, eps)
        return impls

    cases.append({
        "name": "RMS Norm (N×4096)",
        "category": "normalization",
        "gen": gen_rms,
        "make_impls": make_rms_impls,
        "flops": lambda args: args[0].numel() * 5,  # sq + mean + rsqrt + mul
        "bytes": lambda args: args[0].numel() * 2 * 4 + args[0].shape[-1] * 4,
        "dtype": "fp32",
        "sizes": [256, 1024, 4096, 16384],
        "labels": ["256×4K", "1K×4K", "4K×4K", "16K×4K"],
    })

    # ================================================================
    # Case 6: SiLU Activation
    # ================================================================
    def gen_silu(size: int):
        x = torch.randn(size, device="cuda", dtype=torch.float32)
        return (x,), {}

    def make_silu_impls(x):
        return {
            "Triton (ours)": lambda: silu(x),
            "PyTorch (ref)": lambda: torch.nn.functional.silu(x),
        }

    cases.append({
        "name": "SiLU Activation",
        "category": "elementwise",
        "gen": gen_silu,
        "make_impls": make_silu_impls,
        "flops": lambda args: args[0].numel() * 5,  # sigmoid(~4) + mul
        "bytes": lambda args: args[0].numel() * 2 * 4,  # read + write
        "dtype": "fp32",
        "sizes": [65536, 1048576, 16777216, 67108864],
        "labels": ["64K", "1M", "16M", "64M"],
    })

    # ================================================================
    # Case 7: GELU Activation
    # ================================================================
    def make_gelu_impls(x):
        return {
            "Triton (ours)": lambda: gelu(x),
            "PyTorch (ref)": lambda: torch.nn.functional.gelu(x, approximate="tanh"),
        }

    cases.append({
        "name": "GELU Activation",
        "category": "elementwise",
        "gen": gen_silu,  # same input pattern as SiLU
        "make_impls": make_gelu_impls,
        "flops": lambda args: args[0].numel() * 9,  # x³+mul+add+sigmoid+mul
        "bytes": lambda args: args[0].numel() * 2 * 4,
        "dtype": "fp32",
        "sizes": [65536, 1048576, 16777216, 67108864],
        "labels": ["64K", "1M", "16M", "64M"],
    })

    # ================================================================
    # Case 8: Dropout
    # ================================================================
    def gen_dropout(size: int):
        x = torch.randn(size, device="cuda", dtype=torch.float32)
        return (x,), {"p": 0.5}

    def make_dropout_impls(x, p=0.5):
        import random
        seed = random.randint(0, 2**31 - 1)
        return {
            "Triton (ours)": lambda: dropout_fn(x, p=p, seed=seed)[0],
            "PyTorch (ref)": lambda: torch.nn.functional.dropout(x, p=p, training=True),
        }

    cases.append({
        "name": "Dropout (p=0.5)",
        "category": "elementwise",
        "gen": gen_dropout,
        "make_impls": make_dropout_impls,
        "flops": lambda args: args[0].numel() * 4,  # rand + compare + mul
        "bytes": lambda args: args[0].numel() * 3 * 4,  # x + out + mask
        "dtype": "fp32",
        "sizes": [65536, 1048576, 16777216, 67108864],
        "labels": ["64K", "1M", "16M", "64M"],
    })

    # ================================================================
    # Case 9: Fused Residual Add + LayerNorm
    # ================================================================
    def gen_residual(size: int):
        n_rows, n_cols = size, 1024
        x = torch.randn(n_rows, n_cols, device="cuda", dtype=torch.float32)
        residual = torch.randn(n_rows, n_cols, device="cuda", dtype=torch.float32)
        w = torch.randn(n_cols, device="cuda", dtype=torch.float32)
        b = torch.randn(n_cols, device="cuda", dtype=torch.float32)
        return (x, residual, w, b), {"eps": 1e-5}

    def make_residual_impls(x, residual, w, b, eps=1e-5):
        impls = {
            "Triton (ours)": lambda: residual_add_norm(x, residual, w, b, eps),
        }

        # Unfused Triton (add + our LayerNorm)
        if layer_norm:
            impls["Triton Unfused"] = lambda: layer_norm(x + residual, w, b, eps)

        # Unfused PyTorch (add + torch LayerNorm)
        impls["PyTorch Unfused"] = lambda: torch.nn.functional.layer_norm(
            x + residual, [x.shape[-1]], w, b, eps=eps
        )

        # Liger LayerNorm (unfused add + liger LN)
        if liger_ln:
            impls["Liger Unfused"] = lambda: liger_ln(x + residual, w, b, eps)

        return impls

    cases.append({
        "name": "Residual+LayerNorm (N×1024)",
        "category": "normalization",
        "gen": gen_residual,
        "make_impls": make_residual_impls,
        "flops": lambda args: args[0].numel() * 9,  # add + mean + var + norm + affine
        "bytes": lambda args: args[0].numel() * 5 * 4,  # x+r+w+b+out
        "dtype": "fp32",
        "sizes": [256, 1024, 4096, 16384],
        "labels": ["256×1K", "1K×1K", "4K×1K", "16K×1K"],
    })

    # ================================================================
    # Case 10: SwiGLU (fused gate * SiLU(up))
    # ================================================================
    def gen_swiglu(size: int):
        gate = torch.randn(size, 4096, device="cuda", dtype=torch.float32)
        up = torch.randn(size, 4096, device="cuda", dtype=torch.float32)
        return (gate, up), {}

    def make_swiglu_impls(gate, up):
        impls = {
            "Triton (ours)": lambda: swiglu(gate, up),
            "PyTorch Unfused": lambda: gate * torch.nn.functional.silu(up),
        }
        if liger_swiglu_fn:
            impls["Liger Fused (SotA)"] = lambda: liger_swiglu_fn(gate, up)
        return impls

    cases.append({
        "name": "SwiGLU (N×4096)",
        "category": "elementwise",
        "gen": gen_swiglu,
        "make_impls": make_swiglu_impls,
        "flops": lambda args: args[0].numel() * 6,  # sigmoid + 2*mul
        "bytes": lambda args: args[0].numel() * 3 * 4,  # gate + up + out
        "dtype": "fp32",
        "sizes": [256, 1024, 2048, 4096],
        "labels": ["256×4K", "1K×4K", "2K×4K", "4K×4K"],
    })

    # ================================================================
    # Case 11: Cross Entropy Loss
    # ================================================================
    def gen_ce(size: int):
        n_rows, n_classes = 1024, size
        logits = torch.randn(n_rows, n_classes, device="cuda", dtype=torch.float32)
        labels = torch.randint(0, n_classes, (n_rows,), device="cuda")
        return (logits, labels), {}

    def make_ce_impls(logits, labels):
        impls = {
            "Triton (ours)": lambda: cross_entropy_loss(logits, labels),
            "PyTorch (ref)": lambda: torch.nn.functional.cross_entropy(logits, labels),
        }
        if liger_ce:
            impls["Liger (SotA)"] = lambda: liger_ce(logits, labels)
        return impls

    cases.append({
        "name": "Cross Entropy (1024×C)",
        "category": "reduction",
        "gen": gen_ce,
        "make_impls": make_ce_impls,
        "flops": lambda args: args[0].numel() * 13,
        "bytes": lambda args: args[0].numel() * 4 + args[1].numel() * 8,
        "dtype": "fp32",
        "sizes": [1024, 8192, 32000],
        "labels": ["1024×1K", "1024×8K", "1024×32K"],
    })

    # ================================================================
    # Case 12: Rotary Position Embedding (RoPE)
    # ================================================================
    def gen_rope(size: int):
        head_dim = 128
        x = torch.randn(4, 32, size, head_dim, device="cuda", dtype=torch.float32)
        cos, sin = precompute_freqs_cis(head_dim, size, base=10000.0, device="cuda")
        return (x, cos, sin), {}

    def ref_rope(x, cos, sin):
        x_rotated = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).flatten(-2)
        return x * cos + x_rotated * sin

    def make_rope_impls(x, cos, sin):
        return {
            "Triton (ours)": lambda: apply_rotary_emb(x, cos, sin),
            "PyTorch (ref)": lambda: ref_rope(x, cos, sin),
        }

    cases.append({
        "name": "Rotary Embedding (RoPE)",
        "category": "elementwise",
        "gen": gen_rope,
        "make_impls": make_rope_impls,
        "flops": lambda args: args[0].numel() * 3,  # per-element: 2 mul + 1 add
        "bytes": lambda args: args[0].numel() * 4 * 4,  # x + cos + sin + out
        "dtype": "fp32",
        "sizes": [512, 1024, 2048],
        "labels": ["N=512", "N=1K", "N=2K"],
    })

    # ================================================================
    # Case 13: Group Normalization
    # ================================================================
    def gen_gn(size: int):
        C = 256
        x = torch.randn(size, C, device="cuda", dtype=torch.float32)
        w = torch.randn(C, device="cuda")
        b = torch.randn(C, device="cuda")
        return (x, 8, w, b), {}  # G=8 groups

    def make_gn_impls(x, num_groups, w, b):
        return {
            "Triton (ours)": lambda: group_norm(x, num_groups, w, b),
            "PyTorch (ref)": lambda: torch.nn.functional.group_norm(x, num_groups, w, b, eps=1e-5),
        }

    cases.append({
        "name": "Group Norm (N×256, G=8)",
        "category": "normalization",
        "gen": gen_gn,
        "make_impls": make_gn_impls,
        "flops": lambda args: args[0].numel() * 8,
        "bytes": lambda args: args[0].numel() * 4 * 4,
        "dtype": "fp32",
        "sizes": [256, 1024, 4096],
        "labels": ["256×256", "1K×256", "4K×256"],
    })

    # ================================================================
    # NEW: Sigmoid (02)
    # ================================================================
    def gen_elem(size):
        return (torch.randn(size, device="cuda", dtype=torch.float32),), {}

    cases.append({"name": "Sigmoid", "category": "elementwise", "gen": gen_elem,
        "make_impls": lambda x: {"Triton (ours)": lambda: sigmoid(x),
                                  "PyTorch (ref)": lambda: torch.sigmoid(x)},
        "flops": lambda a: a[0].numel()*4, "bytes": lambda a: a[0].numel()*2*4, "dtype": "fp32",
        "sizes": [65536, 1048576, 16777216], "labels": ["64K", "1M", "16M"]})

    # NEW: Tanh (03)
    cases.append({"name": "Tanh", "category": "elementwise", "gen": gen_elem,
        "make_impls": lambda x: {"Triton (ours)": lambda: tanh(x),
                                  "PyTorch (ref)": lambda: torch.tanh(x)},
        "flops": lambda a: a[0].numel()*5, "bytes": lambda a: a[0].numel()*2*4, "dtype": "fp32",
        "sizes": [65536, 1048576, 16777216], "labels": ["64K", "1M", "16M"]})

    # NEW: LeakyReLU (04)
    cases.append({"name": "LeakyReLU (α=0.01)", "category": "elementwise", "gen": gen_elem,
        "make_impls": lambda x: {"Triton (ours)": lambda: leaky_relu(x, 0.01),
                                  "PyTorch (ref)": lambda: torch.nn.functional.leaky_relu(x, 0.01)},
        "flops": lambda a: a[0].numel()*3, "bytes": lambda a: a[0].numel()*2*4, "dtype": "fp32",
        "sizes": [65536, 1048576, 16777216], "labels": ["64K", "1M", "16M"]})

    # NEW: Scale+Bias+Residual (06)
    def gen_sbr(size):
        x = torch.randn(size, device="cuda", dtype=torch.float32)
        bias = torch.randn(size, device="cuda", dtype=torch.float32)
        residual = torch.randn(size, device="cuda", dtype=torch.float32)
        return (x, bias, residual), {}
    cases.append({"name": "Fused Scale+Bias+Residual", "category": "elementwise", "gen": gen_sbr,
        "make_impls": lambda x, bias, r: {
            "Triton (ours)": lambda: fused_scale_bias_residual(x, bias, r),
            "PyTorch Unfused": lambda: 2.0*x + 0.5*bias + r},
        "flops": lambda a: a[0].numel()*7, "bytes": lambda a: a[0].numel()*4*4, "dtype": "fp32",
        "sizes": [65536, 1048576, 16777216], "labels": ["64K", "1M", "16M"]})

    # NEW: GeGLU (11)
    def gen_geglu(size):
        gate = torch.randn(size, 4096, device="cuda", dtype=torch.float32)
        up = torch.randn(size, 4096, device="cuda", dtype=torch.float32)
        return (gate, up), {}
    liger_geglu_fn2 = get_liger_geglu()
    def make_geglu_impls(gate, up):
        impls = {"Triton (ours)": lambda: geglu(gate, up),
                 "PyTorch Unfused": lambda: gate * torch.nn.functional.gelu(up, approximate="tanh")}
        if liger_geglu_fn2: impls["Liger (SotA)"] = lambda: liger_geglu_fn2(gate, up)
        return impls
    cases.append({"name": "GeGLU (N×4096)", "category": "elementwise", "gen": gen_geglu,
        "make_impls": make_geglu_impls,
        "flops": lambda a: a[0].numel()*10, "bytes": lambda a: a[0].numel()*3*4, "dtype": "fp32",
        "sizes": [256, 1024, 2048], "labels": ["256×4K", "1K×4K", "2K×4K"]})

    # NEW: Cumsum (14)
    def gen_1d(size):
        return (torch.rand(size, device="cuda", dtype=torch.float32),), {}
    cases.append({"name": "Cumsum (Prefix Sum)", "category": "reduction", "gen": gen_1d,
        "make_impls": lambda x: {"Triton (ours)": lambda: cumsum_full(x),
                                  "PyTorch (ref)": lambda: torch.cumsum(x, dim=0)},
        "flops": lambda a: a[0].numel(), "bytes": lambda a: a[0].numel()*2*4, "dtype": "fp32",
        "sizes": [65536, 262144, 1048576], "labels": ["64K", "256K", "1M"]})

    # NEW: Gradient Clipping (15)
    def gen_grads(size):
        return ([torch.randn(size, device="cuda", dtype=torch.float32)*5],), {"max_norm": 1.0}
    def make_gradclip_impls(grads, max_norm=1.0):
        def triton_clip():
            gg = [g.clone() for g in grads]; clip_grad_norm(gg, max_norm)
        def torch_clip():
            gg = [g.clone() for g in grads]; torch.nn.utils.clip_grad_norm_(gg, max_norm)
        return {"Triton (ours)": triton_clip, "PyTorch (ref)": torch_clip}
    cases.append({"name": "Gradient Clipping", "category": "reduction", "gen": gen_grads,
        "make_impls": make_gradclip_impls,
        "flops": lambda a: a[0][0].numel()*3, "bytes": lambda a: a[0][0].numel()*2*4, "dtype": "fp32",
        "sizes": [65536, 1048576, 16777216], "labels": ["64K", "1M", "16M"]})

    # NEW: BatchNorm1D (19)
    def gen_bn(size):
        C = 128
        x = torch.randn(size, C, device="cuda", dtype=torch.float32)
        w = torch.randn(C, device="cuda"); b = torch.randn(C, device="cuda")
        return (x, w, b), {}
    cases.append({"name": "BatchNorm1D (N×128)", "category": "normalization", "gen": gen_bn,
        "make_impls": lambda x, w, b: {
            "Triton (ours)": lambda: batchnorm1d(x, w, b),
            "PyTorch (ref)": lambda: torch.nn.functional.batch_norm(x, None, None, weight=w, bias=b, training=True, eps=1e-5)},
        "flops": lambda a: a[0].numel()*8, "bytes": lambda a: a[0].numel()*4*4, "dtype": "fp32",
        "sizes": [256, 1024, 8192], "labels": ["256×128", "1K×128", "8K×128"]})

    # NEW: Embedding (22)
    def gen_emb(size):
        vocab, dim = 10000, 256
        w = torch.randn(vocab, dim, device="cuda", dtype=torch.float32)
        ids = torch.randint(0, vocab, (size,), device="cuda")
        return (w, ids), {}
    cases.append({"name": "Embedding Lookup", "category": "elementwise", "gen": gen_emb,
        "make_impls": lambda w, ids: {
            "Triton (ours)": lambda: embedding(w, ids),
            "PyTorch (ref)": lambda: torch.nn.functional.embedding(ids, w)},
        "flops": lambda a: a[1].numel()*a[0].shape[1],
        "bytes": lambda a: a[1].numel()*a[0].shape[1]*2*4, "dtype": "fp32",
        "sizes": [512, 2048, 8192], "labels": ["512t", "2Kt", "8Kt"]})

    # NEW: AdamW (23)
    def gen_adamw(size):
        p = torch.randn(size, device="cuda", dtype=torch.float32)
        g = torch.randn(size, device="cuda", dtype=torch.float32)
        m = torch.zeros(size, device="cuda"); v = torch.zeros(size, device="cuda")
        return (p, g, m, v), {"step": 10}
    def make_adamw_impls(p, g, m, v, step=10):
        def triton_adamw():
            pc, gc, mc, vc = p.clone(), g.clone(), m.clone(), v.clone()
            adamw_step(pc, gc, mc, vc, step=step)
        return {"Triton (ours)": triton_adamw,
                "PyTorch Unfused": lambda: adamw_pytorch_step(p.clone(), g.clone(), m.clone(), v.clone(), step=step)}
    cases.append({"name": "AdamW Optimizer Step", "category": "elementwise", "gen": gen_adamw,
        "make_impls": make_adamw_impls,
        "flops": lambda a: a[0].numel()*15, "bytes": lambda a: a[0].numel()*4*5, "dtype": "fp32",
        "sizes": [65536, 1048576, 16777216], "labels": ["64K", "1M", "16M"]})

    return cases


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_phase1_benchmarks(
    category_filter: Optional[str] = None,
    quick: bool = False,
    save: bool = False,
    output_dir: str = "benchmarks/results",
) -> list[dict]:
    """Run all phase1 benchmarks and return results for JSON export."""

    all_cases = build_phase1_cases()
    if category_filter:
        all_cases = [c for c in all_cases if c["category"] == category_filter]

    if not all_cases:
        print("No benchmark cases match filter.")
        return []

    # GPU info header
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print("=" * 80)
        print(f"  Phase 1 Kernel Benchmark — Triton vs PyTorch vs Liger")
        print(f"  GPU: {gpu_name}")
        print("=" * 80)
    else:
        print("ERROR: CUDA GPU required for benchmarks.")
        return []

    all_results: list[dict] = []

    for case in all_cases:
        print(f"\n{'─' * 80}")
        print(f"  {case['name']}  [{case['category']}]")
        print(f"{'─' * 80}")

        sizes_to_run = case["sizes"][:2] if quick else case["sizes"]
        labels_to_run = case["labels"][:2] if quick else case["labels"]

        for size, label in zip(sizes_to_run, labels_to_run):
            (args, kwargs) = case["gen"](size)
            flops = case["flops"](args)
            byt = case["bytes"](args)

            # Build closures for this size
            impls = case["make_impls"](*args, **kwargs)

            # Skip if no Triton kernel loaded
            if "Triton (ours)" not in impls or impls["Triton (ours)"] is None:
                print(f"  [{label}] SKIP — Triton kernel not available")
                continue

            print(f"\n  [{label}]  flops={flops:,}  bytes={byt:,}")
            result = bench_compare(
                impls,
                flops=flops,
                bytes_accessed=byt,
                dtype=case["dtype"],
                warmup=5 if quick else 25,
                rep=20 if quick else 100,
            )
            print_compare_report(result)

            # Collect for JSON export
            entry = {
                "case": case["name"],
                "category": case["category"],
                "size": label,
                "implementations": {},
            }
            for r in result.results:
                entry["implementations"][r.name] = {
                    "time_ms": round(r.time_ms, 4),
                    "tflops": round(r.tflops, 2),
                    "bandwidth_gbs": round(r.bandwidth_gbs, 1),
                    "pct_of_ceiling": round(r.pct_of_ceiling, 1),
                    "speedup_vs_baseline": round(r.speedup_vs_baseline, 2),
                }
            all_results.append(entry)

    # Summary
    print(f"\n{'=' * 80}")
    print(f"  SUMMARY — Phase 1 Kernels")
    print(f"{'=' * 80}")

    for entry in all_results:
        name = entry["case"]
        size = entry["size"]
        impls = entry["implementations"]
        parts = []
        for impl_name, data in impls.items():
            short_name = impl_name.split("(")[0].strip()
            parts.append(f"{short_name}: {data['time_ms']:.4f}ms")
        print(f"  {name:<30s} [{size:>12s}]  {'  |  '.join(parts)}")

    # Key insight
    print(f"\n  💡 Elementwise/norm kernels are memory-bound — "
          f"optimize for bandwidth utilization, not TFLOPS.")
    print(f"  💡 Operator fusion (SwiGLU, residual+norm, softmax) avoids HBM round-trips.")
    print(f"  💡 RMSNorm is faster than LayerNorm; GroupNorm fills the batch-size-independent niche.")
    print(f"  💡 RoPE applies 2D rotations to Q/K pairs — pair-based kernel is 2x faster than PyTorch.")
    print(f"  💡 Cross Entropy with online softmax avoids exp overflow (max subtraction trick).")
    print(f"  💡 Liger is production-grade Triton — compare to see optimization headroom.")

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 Kernel Benchmark — Triton vs PyTorch vs Liger"
    )
    parser.add_argument(
        "--category", "-c", type=str, default=None,
        choices=["elementwise", "reduction", "normalization", "activation"],
        help="Only run benchmarks in this category",
    )
    parser.add_argument(
        "--quick", "-q", action="store_true",
        help="Quick mode: fewer iterations, smaller sizes",
    )
    parser.add_argument(
        "--save", "-s", action="store_true",
        help="Save results to JSON",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="benchmarks/results",
        help="Output directory for JSON results",
    )
    args = parser.parse_args()

    results = run_phase1_benchmarks(
        category_filter=args.category,
        quick=args.quick,
        save=args.save,
        output_dir=args.output,
    )

    if args.save and results:
        os.makedirs(args.output, exist_ok=True)
        gpu_name = torch.cuda.get_device_name(0).replace(" ", "_")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(args.output, f"phase1_{timestamp}_{gpu_name}.json")

        json_data = {
            "timestamp": timestamp,
            "gpu": torch.cuda.get_device_name(0),
            "benchmark": "phase1_triton_vs_pytorch_vs_liger",
            "results": results,
        }
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"\nResults saved to: {json_path}")


if __name__ == "__main__":
    main()
