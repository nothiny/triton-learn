"""
bench_attention_triton_vs_sota.py — Triton Attention vs SOTA implementations

对比:
  1. Triton Flash Attention v1 (12_flash_attention_v1)
  2. Triton Flash Attention v2 (13_flash_attention_v2, causal)
  3. Triton GQA (15_grouped_query_attention)
  4. Triton Sliding Window (16_sliding_window_attention)
  5. Triton Attention Bias / ALiBi (17_attention_bias)
  6. PyTorch SDPA (torch.nn.functional.scaled_dot_product_attention)
     → dispatches to cuDNN FlashAttention-2 on H100/A100
  7. flash-attn library (if installed, via `from flash_attn import flash_attn_func`)

flash-attn 库 (Tri Dao 官方实现) 安装方式:
  pip install flash-attn --no-build-isolation
  # 或从源码:
  # cd third_party/flash-attention && pip install -e .

  API 调用示例:
    from flash_attn import flash_attn_func
    output = flash_attn_func(q, k, v, causal=True)

PyTorch SDPA 调用:
  output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
  # H100/A100 上自动 dispatch 到 cuDNN FlashAttention-2

运行: python benchmarks/bench_attention_triton_vs_sota.py
"""

import sys
import math
from pathlib import Path

import torch
import triton
from triton.testing import do_bench

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_kernel(module_path: str, fn_name: str):
    """通过 importlib 动态加载 kernel 函数。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(fn_name, module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, fn_name)


# ─── flash-attn library (if available) ───────────────────────────────────────
try:
    from flash_attn import flash_attn_func as _fa_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    _fa_func = None


def flash_attn_ref(q, k, v, causal=False):
    """调用 Tri Dao 的 flash-attention 库。

    注意 layout 转换:
      Triton: (batch, nheads, seqlen, headdim)
      flash-attn: (batch, seqlen, nheads, headdim)
    """
    if not HAS_FLASH_ATTN:
        raise RuntimeError("flash-attn not installed. Run: pip install flash-attn")
    # Triton layout → flash-attn layout
    q_fa = q.transpose(1, 2).contiguous()
    k_fa = k.transpose(1, 2).contiguous()
    v_fa = v.transpose(1, 2).contiguous()
    out_fa = _fa_func(q_fa, k_fa, v_fa, causal=causal)
    # flash-attn layout → Triton layout
    return out_fa.transpose(1, 2).contiguous()


# ─── PyTorch SDPA reference ──────────────────────────────────────────────────
def torch_sdpa(q, k, v, causal=False):
    """PyTorch SDPA — H100 上 dispatch 到 cuDNN FlashAttention-2。

    注意: PyTorch 2.11 SDPA 不支持 GQA 的 head broadcasting。
    对 GQA 场景，需要先 expand K/V 到匹配 Q 的 head 数。
    """
    if q.size(1) != k.size(1):
        # GQA: expand K/V heads to match Q heads
        n_groups = q.size(1) // k.size(1)
        k = k.repeat_interleave(n_groups, dim=1)
        v = v.repeat_interleave(n_groups, dim=1)
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=causal,
    )


# ─── Benchmark helpers ───────────────────────────────────────────────────────
def bench_case(name, fn, warmup=25, rep=100):
    """运行 benchmark 并返回 (name, ms) tuple。"""
    ms = do_bench(fn, warmup=warmup, rep=rep)
    return name, ms


def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_result(name, ms, ref_ms=None, extra=""):
    """打印单行结果。"""
    if ref_ms:
        speedup = ref_ms / ms
        print(f"  {name:<35s}: {ms:8.4f}ms  ({speedup:.2f}x vs ref)  {extra}")
    else:
        print(f"  {name:<35s}: {ms:8.4f}ms  {extra}")


# ==============================================================================
# Main
# ==============================================================================


def main():
    print("=" * 70)
    print("  Attention Benchmark: Triton vs SOTA")
    print("=" * 70)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"  GPU: {gpu_name}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  flash-attn: {'installed' if HAS_FLASH_ATTN else 'NOT installed'}")
    print(f"  Triton: {triton.__version__}")
    print()

    # ==========================================================================
    # 1. Standard Attention (non-causal)
    # ==========================================================================
    print_header("1. Standard Attention (non-causal)")

    configs_standard = [
        # (batch, n_heads, seq_len, d_head)
        (1, 8, 128, 64),
        (1, 8, 256, 64),
        (1, 8, 512, 64),
        (1, 16, 256, 128),
        (2, 16, 512, 64),
    ]

    for B, H, N, D in configs_standard:
        q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

        print(f"\n  [{B}×{H}×{N}×{D}]:")

        # Triton v1
        fa1_fn = load_kernel("phase2_compute/12_flash_attention_v1.py",
                             "flash_attention_v1")
        ms_triton_v1 = do_bench(lambda: fa1_fn(q, k, v))

        # PyTorch SDPA
        ms_sdpa = do_bench(lambda: torch_sdpa(q, k, v, causal=False))

        # flash-attn (if installed)
        if HAS_FLASH_ATTN:
            ms_fa = do_bench(lambda: flash_attn_ref(q, k, v, causal=False))

        tflops = (4 * B * H * N * N * D) / (ms_sdpa * 1e-3) / 1e12
        print(f"    Triton v1:     {ms_triton_v1:.4f}ms "
              f"({ms_triton_v1/ms_sdpa:.2f}x SDPA)")
        print(f"    SDPA (cuDNN):  {ms_sdpa:.4f}ms  (baseline)")
        if HAS_FLASH_ATTN:
            print(f"    flash-attn:    {ms_fa:.4f}ms "
                  f"({ms_fa/ms_sdpa:.2f}x SDPA)")

    # ==========================================================================
    # 2. Causal Attention
    # ==========================================================================
    print_header("2. Causal Attention (autoregressive)")

    configs_causal = [
        (1, 8, 128, 64),
        (1, 8, 256, 64),
        (1, 8, 512, 64),
        (1, 8, 1024, 64),
        (1, 16, 256, 128),
        (2, 16, 512, 64),
    ]

    for B, H, N, D in configs_causal:
        q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

        print(f"\n  [{B}×{H}×{N}×{D}]:")

        # Triton v2 (causal)
        fa2_fn = load_kernel("phase2_compute/13_flash_attention_v2.py",
                             "flash_attention_v2")
        ms_triton_v2 = do_bench(lambda: fa2_fn(q, k, v, causal=True))

        # PyTorch SDPA (causal)
        ms_sdpa = do_bench(lambda: torch_sdpa(q, k, v, causal=True))

        if HAS_FLASH_ATTN:
            ms_fa = do_bench(lambda: flash_attn_ref(q, k, v, causal=True))

        tflops = (2 * B * H * N * N * D) / (ms_sdpa * 1e-3) / 1e12
        print(f"    Triton v2:     {ms_triton_v2:.4f}ms "
              f"({ms_triton_v2/ms_sdpa:.2f}x SDPA)")
        print(f"    SDPA (cuDNN):  {ms_sdpa:.4f}ms  (baseline)")
        if HAS_FLASH_ATTN:
            print(f"    flash-attn:    {ms_fa:.4f}ms "
                  f"({ms_fa/ms_sdpa:.2f}x SDPA)")

    # ==========================================================================
    # 3. Grouped Query Attention (GQA)
    # ==========================================================================
    print_header("3. Grouped Query Attention (GQA)")

    configs_gqa = [
        # (B, n_heads_q, n_heads_kv, N, D, groups, causal)
        (1, 8, 2, 128, 64, 4, False, "4 groups, non-causal"),
        (1, 8, 2, 256, 64, 4, True,  "4 groups, causal"),
        (1, 8, 1, 128, 64, 8, True,  "MQA (1 KV head)"),
        (2, 32, 8, 128, 64, 4, True, "Llama-2 style"),
    ]

    gqa_fn = load_kernel("phase2_compute/15_grouped_query_attention.py",
                         "grouped_query_attention")

    for cfg in configs_gqa:
        B, Hq, Hkv, N, D, groups, causal, desc = cfg
        q = torch.randn(B, Hq, N, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, Hkv, N, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, Hkv, N, D, device="cuda", dtype=torch.float16)

        print(f"\n  {desc} [{B}×{Hq}Q/{Hkv}KV×{N}×{D}]:")

        ms_triton = do_bench(lambda: gqa_fn(q, k, v, causal=causal))

        # SDPA with GQA (PyTorch 2.1+)
        ms_sdpa = do_bench(lambda: torch_sdpa(q, k, v, causal=causal))

        print(f"    Triton GQA:    {ms_triton:.4f}ms "
              f"({ms_triton/ms_sdpa:.2f}x SDPA)")
        print(f"    SDPA:          {ms_sdpa:.4f}ms  (baseline)")

    # ==========================================================================
    # 4. Sliding Window Attention
    # ==========================================================================
    print_header("4. Sliding Window Attention")

    sw_fn = load_kernel("phase2_compute/16_sliding_window_attention.py",
                        "sliding_window_attention")

    configs_sw = [
        # (B, H, N, D, window, causal)
        (1, 8, 128, 64, 32, True, "small window"),
        (1, 8, 256, 64, 64, True, "medium window"),
        (1, 8, 512, 64, 128, True, "large window"),
        (1, 8, 512, 64, 32, True, "small/long seq"),
    ]

    for cfg in configs_sw:
        B, H, N, D, window, causal, desc = cfg
        q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

        print(f"\n  {desc} [{B}×{H}×{N}×{D} win={window}]:")

        ms_triton = do_bench(lambda: sw_fn(q, k, v, window, causal=causal))

        # SDPA doesn't natively support sliding window in PyTorch 2.x API
        # Use flash-attn if available, else skip
        if HAS_FLASH_ATTN:
            ms_fa = do_bench(
                lambda: flash_attn_ref(q, k, v, causal=causal))
            # Note: flash_attn_func doesn't have window_size in the basic API
            # Use flash_attn_func(q,k,v, window_size=(window, 0)) in newer versions
            print(f"    Triton SW:     {ms_triton:.4f}ms")
            print(f"    flash-attn:    {ms_fa:.4f}ms (full causal, no window)")
        else:
            ms_sdpa = do_bench(lambda: torch_sdpa(q, k, v, causal=causal))
            print(f"    Triton SW:     {ms_triton:.4f}ms")
            print(f"    SDPA (full):   {ms_sdpa:.4f}ms  (full causal, no window)")

    # ==========================================================================
    # 5. Attention with ALiBi Bias (skipped due to API incompatibility)
    # ==========================================================================
    print_header("5. Attention with ALiBi Bias")
    print("  ⚠️  Skipped: build_alibi_bias has device compatibility issue.")
    print("       Run phase2_compute/17_attention_bias.py directly instead.")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print_header("Summary")

    print("""
  基准线: PyTorch SDPA (内部 dispatch 到 cuDNN FlashAttention-2)
  Triton kernels 是教学实现，没有达到生产级性能，主要差距:
    1. Tiling 策略: 手工分块 vs cuDNN 自动调优
    2. Shared memory: Triton 编译器自动管理 vs 手工优化
    3. 寄存器使用: 未做 warp specialization
    4. Memory coalescing: 基本的合并访问 vs 最优 pattern

  flash-attn 库 (Tri Dao):
    提供了更接近硬件的优化，包括 warp specialization、TMA (Hopper)、
    2-CTA cooperative groups 等。安装后可补充对比。

  安装 flash-attn:
    pip install flash-attn --no-build-isolation

  调用方式:
    from flash_attn import flash_attn_func
    output = flash_attn_func(q, k, v, causal=True)
    # q, k, v: (batch, seqlen, nheads, headdim) — 注意 layout 不同！
    # Triton 用 (batch, nheads, seqlen, headdim)
    # flash-attn 默认用 (batch, seqlen, nheads, headdim)
""")


if __name__ == "__main__":
    main()
