.PHONY: help run-phase1 run-phase2 dump-ir profile-matmul profile-flash test test-gpu check-env clean

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ============================================================
# Environment
# ============================================================

check-env:  ## Verify Triton + PyTorch + CUDA installation
	@echo "=== Environment Check ==="
	@python -c "\
import triton; \
import torch; \
print(f'Triton  {triton.__version__}'); \
print(f'PyTorch {torch.__version__}'); \
print(f'CUDA    {torch.version.cuda}'); \
print(f'GPU     {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); \
print(f'GPU count: {torch.cuda.device_count()}')"
	@echo "=== OK ==="

# ============================================================
# Phase 1 — Fundamentals
# ============================================================

run-vector-add:  ## Run vector add kernel
	python phase1_fundamentals/01_vector_add.py

run-softmax:  ## Run fused softmax kernel
	python phase1_fundamentals/02_fused_softmax.py

run-relu-bias:  ## Run fused ReLU+bias kernel
	python phase1_fundamentals/03_fused_relu_bias.py

run-layernorm:  ## Run layer norm kernel
	python phase1_fundamentals/04_layer_norm.py

run-phase1: run-vector-add run-softmax run-relu-bias run-layernorm  ## Run all Phase 1 kernels

# ============================================================
# Phase 2 — Compute
# ============================================================

run-matmul-naive:  ## Run naive matmul
	python phase2_compute/01_matmul_naive.py

run-matmul-tiled:  ## Run tiled matmul (production-grade)
	python phase2_compute/02_matmul_tiled.py

run-matmul-autotuned:  ## Run autotuned matmul
	python phase2_compute/03_matmul_autotuned.py

run-flash-v1:  ## Run Flash Attention v1
	python phase2_compute/04_flash_attention_v1.py

run-flash-v2:  ## Run Flash Attention v2
	python phase2_compute/05_flash_attention_v2.py

run-conv:  ## Run depthwise conv
	python phase2_compute/06_depthwise_conv.py

run-phase2: run-matmul-naive run-matmul-tiled run-matmul-autotuned  ## Run all Phase 2 kernels

# ============================================================
# Benchmarking
# ============================================================

bench:  ## Run all benchmarks (quick mode)
	python benchmarks/bench_runner.py -q

bench-full:  ## Run full benchmarks
	python benchmarks/bench_runner.py

bench-gemm:  ## Benchmark GEMM kernels only
	python benchmarks/bench_runner.py -c gemm

bench-attention:  ## Benchmark attention kernels only
	python benchmarks/bench_runner.py -c attention

bench-profile:  ## Run benchmarks with torch.profiler + chrome traces
	python benchmarks/bench_runner.py --profile --trace-out traces/

bench-json:  ## Export benchmark results to JSON
	python benchmarks/bench_runner.py -j reports/bench_results.json

# --- Standalone tiered benchmarks ---

bench-matmul:  ## GEMM benchmark: Triton vs cuBLAS vs roofline
	python benchmarks/bench_matmul.py --save

bench-attn:  ## Attention benchmark: Flash Attn vs SDPA vs naive
	python benchmarks/bench_attention.py --save

bench-elem:  ## Elementwise/norm benchmark: Triton vs Liger vs PyTorch
	python benchmarks/bench_elementwise.py --save

bench-all: check-gpu  ## All standalone benchmarks + save results
	python benchmarks/bench_matmul.py --save && \
	python benchmarks/bench_attention.py --save && \
	python benchmarks/bench_elementwise.py --save

check-gpu:  ## Print GPU hardware specification
	python benchmarks/hardware_spec.py

diff-results:  ## Compare two benchmark result JSON files
	python benchmarks/compare_results.py benchmarks/results/

# --- Optional SotA dependencies ---

install-flash-attn:  ## Install flash-attn (Tri Dao's official implementation)
	pip install flash-attn --no-build-isolation

install-liger:  ## Install liger-kernel (LinkedIn's Triton kernel library)
	pip install liger-kernel

# ============================================================
# Profiling (requires ncu / nsys)
# ============================================================

profile-matmul:  ## Profile tiled matmul with ncu
	ncu --set full -o reports/matmul_tiled python phase2_compute/02_matmul_tiled.py

profile-flash:  ## Profile Flash Attention v2 with ncu
	ncu --set full -o reports/flash_attn_v2 python phase2_compute/05_flash_attention_v2.py

# ============================================================
# Phase 3 — Compiler Internals
# ============================================================

dump-ir:  ## Dump Triton IR stages (TTIR → TTGIR → LLVM → PTX)
	python phase3_compiler/01_dump_ir.py

layout-analysis:  ## Analyze layout encodings
	python phase3_compiler/02_layout_analysis.py

ptx-analysis:  ## Read and annotate generated PTX
	python phase3_compiler/04_ptx_analysis.py

# ============================================================
# Testing
# ============================================================

test:  ## Run all tests (CPU-only)
	pytest tests/ -v -k "not gpu"

test-gpu:  ## Run GPU tests
	pytest tests/ -v -m gpu

test-all:  ## Run all tests
	pytest tests/ -v

# ============================================================
# Utilities
# ============================================================

clean:  ## Remove caches and generated files
	rm -rf __pycache__ phase1_fundamentals/__pycache__ phase2_compute/__pycache__
	rm -rf phase3_compiler/__pycache__ utils/__pycache__ tests/__pycache__
	rm -rf .pytest_cache triton_cache/
	rm -rf reports/
	find . -name "*.ptx" -delete
	find . -name "*.ll" -delete

reports:
	mkdir -p reports
