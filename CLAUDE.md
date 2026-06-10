# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A systematic GPU kernel learning project using **Triton** as the main programming model, with side exploration of Triton's compiler internals and CUTILE/CUTLASS. Target audience: developers with compiler backend experience (LLVM, register allocation, SSA IR) learning GPU programming.

The project is structured in 4 sequential phases, each with kernel files in `phase{N}_*/` and companion notes in `notes/`. The kernel at `phase2_compute/02_matmul_tiled.py` and the note `notes/03_triton_compiler_pipeline.md` are the most important files.

## Environment

- Python 3.12, CUDA ≥ 12.0, Triton 3.6.0, PyTorch 2.11.0+cu128
- Package manager: `uv` (see `pyproject.toml` and `uv.lock`)
- Install: `pip install -e ".[dev]"` or `uv pip install -e ".[dev]"`
- Optional SotA refs: `pip install -e ".[sota]"` (flash-attn, liger-kernel, xformers)
- Lint/format: `ruff` (line-length 100, double quotes) and `black` (line-length 100, py310)

## Commands

```bash
make check-env          # Verify Triton + PyTorch + CUDA
make test               # CPU-only tests (pytest -v -k "not gpu")
make test-gpu           # GPU tests (pytest -v -m gpu)
make test-all           # All tests

# Run individual kernels
make run-vector-add     # phase1_fundamentals/01_vector_add.py
make run-softmax        # phase1_fundamentals/02_fused_softmax.py
make run-matmul-tiled   # phase2_compute/02_matmul_tiled.py
make run-flash-v1       # phase2_compute/04_flash_attention_v1.py

# Compiler IR inspection
make dump-ir            # phase3_compiler/01_dump_ir.py
make layout-analysis    # phase3_compiler/02_layout_analysis.py
make ptx-analysis       # phase3_compiler/04_ptx_analysis.py

# Benchmarks
make bench              # Quick benchmark (all kernels vs PyTorch/cuBLAS/Liger)
make bench-gemm         # GEMM only
make bench-profile      # With torch.profiler + chrome traces
make bench-json         # Export to JSON
make bench-matmul       # Standalone: GEMM three-tier (Triton vs cuBLAS vs roofline)
make bench-attn         # Standalone: Attention three-tier (Flash Attn vs SDPA vs naive)
make bench-elem         # Standalone: Elementwise/norm (Triton vs Liger vs PyTorch)
make bench-all          # All standalone benchmarks + save results
make check-gpu          # Print GPU hardware specification + roofline ridge points

# Profiling (needs NVIDIA Nsight Compute)
make profile-matmul     # ncu --set full on tiled matmul
make clean              # Remove caches, __pycache__, generated IR/PTX, reports/
```

## Architecture

### Kernel pattern

Every kernel file follows the same structure:

1. **`@triton.autotune`** (optional) — sweeps `triton.Config` over `BLOCK_SIZE`, `num_warps`, `num_stages`. The `key` argument names the parameters that determine which config to cache.
2. **`@triton.jit` kernel function** — takes raw pointers, shape integers, strides, and `tl.constexpr` block sizes. Uses `tl.program_id`, `tl.arange`, `tl.load`, `tl.store`, `tl.dot`. Masking with `mask=mask, other=0.0` handles boundary elements.
3. **Python wrapper function** — allocates output, computes `grid` as a `lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)`, launches `kernel[grid](...)`.
4. **`main()`** — correctness check vs PyTorch reference, then CUDA-event-based timing.
5. **`# PERFORMANCE NOTES`** — roofline analysis, bottleneck reasoning, optimization directions.

### Utility modules (`utils/`)

- **`profiler.py`** — `KernelProfiler` class with CUDA-event timing, TFLOPS/bandwidth calculation, roofline bottleneck detection. `GPUInfo.detect()` auto-identifies GPU specs (H100, A100, RTX 4090, etc.). `quick_bench()` for one-liner use. `bench_compare()` benchmarks multiple implementations side-by-side and prints a comparison table with speedup vs baseline.
- **`checker.py`** — `check_allclose()` for numerical verification with detailed error reporting (max abs/rel diff, exceed fraction, worst-element location). Includes PyTorch reference implementations (`ref_softmax`, `ref_layer_norm`, etc.).
- **`ir_dump.py`** — Environment-variable-based IR dumping (`TRITON_KERNEL_DUMP`, `MLIR_PRINT_IR_AFTER_ALL`), PTX annotation helper that labels register declarations, shared memory, MMA instructions, barriers, and global/shared memory access with latency notes.
- **`roofline.py`** — `GPUSpec` dataclass with peak TFLOPS/bandwidth for H100, A100, RTX 4090, etc. `get_gpu_spec()` auto-detects GPU and returns specs. `roofline_analysis()` classifies kernel as compute-bound or memory-bound given FLOPs, bytes_accessed, and time_ms. `print_gpu_spec()` prints a formatted hardware spec table.

### Benchmark system (`benchmarks/`)

- **`hardware_spec.py`** — Standalone script: auto-detects GPU and prints full hardware spec table with roofline ridge points. Run via `make check-gpu`.
- **`bench_cases.py`** — Defines `BenchCase` dataclass (name, category, `triton_fn`, `ref_fn`, `input_gen`, `flops_calc`, `bytes_calc`, sizes, tolerances). `build_cases()` auto-loads Triton kernels from phase1/phase2 via `importlib`. Includes liger-kernel comparisons.
- **`bench_runner.py`** — `BenchRunner` class: correctness check → CUDA-event timing for both Triton and reference → computes TFLOPS/bandwidth → prints comparison table grouped by category.
- **`bench_matmul.py`** — Standalone GEMM three-tier benchmark: scans multiple (M,N,K) sizes, compares Triton vs cuBLAS vs roofline ceiling. Supports `--plot` (TFLOPS vs size), `--profile` (torch.profiler), `--save` (JSON export).
- **`bench_attention.py`** — Standalone attention benchmark: sweeps sequence lengths, compares Flash Attention vs torch SDPA vs naive attention. Reports memory savings (O(N²)→O(N)).
- **`bench_elementwise.py`** — Standalone elementwise/norm benchmark: compares Triton kernels vs Liger Kernel vs PyTorch. All kernels are memory-bound — optimizes for bandwidth utilization, not TFLOPS.
- **`references/`** — SotA reference wrappers with graceful fallback:
  - `cublas_gemm.py` — cuBLAS via `torch.mm`
  - `flash_attn_ref.py` — flash-attn library + torch SDPA + naive attention, with memory analysis utilities
  - `liger_ref.py` — Liger Kernel wrappers for LayerNorm, RMSNorm, SwiGLU, GeGLU, Softmax (handles parameter order differences)

### Test patterns (`tests/`)

- `conftest.py` — session-scoped `device` and `gpu_name` fixtures. Defines `gpu` and `triton` pytest markers. GPU tests are auto-skipped when CUDA is unavailable.
- Tests use `importlib.util.spec_from_file_location` to import functions from numeric-prefixed filenames (e.g., `phase1_fundamentals/01_vector_add`), because `import` can't handle filenames starting with digits.
- `pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), ...)` at module level in test files.
- Markers: `@pytest.mark.slow` for tests >10s, `@pytest.mark.xfail` for known-broken implementations (e.g., LayerNorm simplified 3-pass version).

### Key conventions

- **`tl.constexpr`** parameters are annotated with `# [COMPILER]` comments explaining what the compiler does with them (e.g., unrolls, maps to template parameters).
- **`# PERFORMANCE NOTES`** at the end of each kernel file — roofline analysis with actual FLOP/byte arithmetic intensity, bottleneck classification, and optimization roadmap.
- **GPU semantic comments**: explain what happens at the GPU execution level (coalescing, shared memory banking, warp scheduling), not just what the Python/Triton code does.
- **Autotune `key`**: always the shape parameters (`["n_elements"]` or `["M", "N", "K"]`), never the block sizes — Triton caches one compiled variant per unique key.
- **Grid lambda**: always `lambda meta: (...)` so autotune can pass the selected config's `BLOCK_SIZE` via `meta`.
- **Dtype convention**: fp32 for elementwise/reduction, fp16 for GEMM/attention (to use Tensor Cores). Accumulators in `tl.float32` inside kernels.
