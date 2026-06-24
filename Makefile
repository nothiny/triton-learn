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

# Phase 1 — Group 1: Basics
run-vector-add:  ## Run 01 vector add kernel
	python phase1_fundamentals/01_vector_add.py
run-sigmoid:  ## Run 02 sigmoid kernel
	python phase1_fundamentals/02_sigmoid.py
run-tanh:  ## Run 03 tanh kernel
	python phase1_fundamentals/03_tanh.py

# Phase 1 — Group 2: Elementwise Fusion
run-leaky-relu:  ## Run 04 leaky relu kernel
	python phase1_fundamentals/04_leaky_relu.py
run-relu-bias:  ## Run 05 fused ReLU+bias kernel
	python phase1_fundamentals/05_fused_relu_bias.py
run-scale-bias-residual:  ## Run 06 fused scale+bias+residual kernel
	python phase1_fundamentals/06_fused_scale_bias_residual.py

# Phase 1 — Group 3: Advanced Activations
run-silu:  ## Run 07 SiLU kernel
	python phase1_fundamentals/07_silu.py
run-gelu:  ## Run 08 GELU kernel
	python phase1_fundamentals/08_gelu.py
run-dropout:  ## Run 09 dropout kernel
	python phase1_fundamentals/09_dropout.py

# Phase 1 — Group 4: Gated Activations
run-swiglu:  ## Run 10 SwiGLU kernel
	python phase1_fundamentals/10_swiglu.py
run-geglu:  ## Run 11 GeGLU kernel
	python phase1_fundamentals/11_geglu.py

# Phase 1 — Group 5: Reductions
run-vector-sum:  ## Run 12 vector sum kernel
	python phase1_fundamentals/12_vector_sum.py
run-vector-max:  ## Run 13 vector max kernel
	python phase1_fundamentals/13_vector_max.py
run-vector-norm:  ## Run 14 L2 norm kernel
	python phase1_fundamentals/14_vector_norm_l2.py
run-welford:  ## Run 15 Welford mean+var kernel
	python phase1_fundamentals/15_welford_mean_var.py
run-logsumexp:  ## Run 16 LogSumExp kernel
	python phase1_fundamentals/16_logsumexp.py
run-softmax:  ## Run 17 fused softmax kernel
	python phase1_fundamentals/17_fused_softmax.py
run-cross-entropy:  ## Run 18 cross entropy kernel
	python phase1_fundamentals/18_cross_entropy.py
run-cumsum:  ## Run 19 cumsum kernel
	python phase1_fundamentals/19_cumsum.py
run-grad-clip:  ## Run 20 gradient clipping kernel
	python phase1_fundamentals/20_gradient_clipping.py

# Phase 1 — Group 6: Normalizations
run-layernorm:  ## Run 21 layer norm kernel
	python phase1_fundamentals/21_layer_norm.py
run-rms-norm:  ## Run 22 RMS norm kernel
	python phase1_fundamentals/22_rms_norm.py
run-group-norm:  ## Run 23 group norm kernel
	python phase1_fundamentals/23_group_norm.py
run-batch-norm:  ## Run 24 batch norm kernel
	python phase1_fundamentals/24_batch_norm.py
run-residual-norm:  ## Run 25 residual+norm kernel
	python phase1_fundamentals/25_residual_add_norm.py

# Phase 1 — Group 7: Position / Embedding / Optimizer
run-rope:  ## Run 26 rotary embedding kernel
	python phase1_fundamentals/26_rotary_embedding.py
run-embedding:  ## Run 27 embedding kernel
	python phase1_fundamentals/27_embedding.py
run-adamw:  ## Run 28 AdamW kernel
	python phase1_fundamentals/28_adamw.py
run-mean-var:  ## Run 29 parallel mean+var (E[X²]-E[X]²) kernel
	python phase1_fundamentals/29_parallel_mean_var.py
run-argmax:  ## Run 30 argmax kernel
	python phase1_fundamentals/30_argmax_reduce.py
run-topk:  ## Run 31 topk kernel
	python phase1_fundamentals/31_topk_selection.py
run-mse:  ## Run 32 MSE loss kernel
	python phase1_fundamentals/32_mse_loss.py
run-hinge:  ## Run 33 hinge loss kernel
	python phase1_fundamentals/33_hinge_loss.py
run-l1:  ## Run 34 L1 loss kernel
	python phase1_fundamentals/34_l1_loss.py
run-relu6:  ## Run 35 ReLU6 kernel
	python phase1_fundamentals/35_relu6_clamp.py
run-hard-sigmoid:  ## Run 36 hard sigmoid kernel
	python phase1_fundamentals/36_hard_sigmoid.py
run-hard-swish:  ## Run 37 hard swish kernel
	python phase1_fundamentals/37_hard_swish.py
run-dot:  ## Run 38 dot product kernel
	python phase1_fundamentals/38_vector_dot.py
run-transpose:  ## Run 39 2D transpose kernel
	python phase1_fundamentals/39_transpose_2d.py
run-concat:  ## Run 40 concat kernel
	python phase1_fundamentals/40_concat.py
run-max-pool:  ## Run 41 max pool kernel
	python phase1_fundamentals/41_max_pool1d.py
run-avg-pool:  ## Run 42 avg pool kernel
	python phase1_fundamentals/42_avg_pool1d.py
run-scaled-dot:  ## Run 43 scaled dot product kernel
	python phase1_fundamentals/43_scaled_dot_product.py
run-causal-mask:  ## Run 44 causal mask kernel
	python phase1_fundamentals/44_causal_mask.py
run-one-hot:  ## Run 45 one-hot kernel
	python phase1_fundamentals/45_one_hot.py
run-weight-decay:  ## Run 46 weight decay kernel
	python phase1_fundamentals/46_weight_decay.py
run-ema:  ## Run 47 EMA kernel
	python phase1_fundamentals/47_ema.py
run-cosine-sim:  ## Run 48 cosine similarity kernel
	python phase1_fundamentals/48_cosine_similarity.py
run-gelu-exact:  ## Run 49 exact GELU kernel
	python phase1_fundamentals/49_gelu_accurate.py
run-fused-bias-gelu:  ## Run 50 fused bias+GELU kernel
	python phase1_fundamentals/50_fused_bias_gelu.py

# All Phase 1
run-phase1: run-vector-add run-sigmoid run-tanh run-leaky-relu \
           run-relu-bias run-scale-bias-residual run-silu run-gelu run-dropout \
           run-swiglu run-geglu \
           run-vector-sum run-vector-max run-vector-norm run-welford run-logsumexp \
           run-softmax run-cross-entropy run-cumsum run-grad-clip \
           run-layernorm run-rms-norm run-group-norm run-batch-norm run-residual-norm \
           run-rope run-embedding run-adamw run-mean-var \
           run-argmax run-topk run-mse run-hinge run-l1 run-relu6 \
           run-hard-sigmoid run-hard-swish run-dot run-transpose run-concat \
           run-max-pool run-avg-pool run-scaled-dot run-causal-mask run-one-hot \
           run-weight-decay run-ema run-cosine-sim run-gelu-exact run-fused-bias-gelu  ## Run all 50 Phase 1 kernels

# ============================================================
# Phase 2 — Compute
# ============================================================

# --- MatMul (01-06) ---
run-matmul-naive:  ## Run naive matmul
	python phase2_compute/01_matmul_naive.py

run-matmul-tiled:  ## Run tiled matmul (production-grade)
	python phase2_compute/02_matmul_tiled.py

run-matmul-autotuned:  ## Run autotuned matmul (GROUP_M swizzling)
	python phase2_compute/03_matmul_autotuned.py

run-split-k:  ## Run split-K parallel GEMM
	python phase2_compute/04_matmul_split_k.py

run-fused-matmul:  ## Run fused matmul + bias + activation
	python phase2_compute/05_matmul_fused_bias_act.py

run-transpose-variants:  ## Run 4 transpose variants (NN/NT/TN/TT)
	python phase2_compute/06_matmul_transpose.py

# --- Attention (07-13) ---
run-flash-v1:  ## Run Flash Attention v1
	python phase2_compute/07_flash_attention_v1.py

run-flash-v2:  ## Run Flash Attention v2
	python phase2_compute/08_flash_attention_v2.py

run-conv:  ## Run depthwise conv
	python phase2_compute/09_depthwise_conv.py

run-flash-bwd:  ## Run Flash Attention backward pass
	python phase2_compute/10_flash_attention_backward.py

run-gqa:  ## Run Grouped Query Attention (GQA)
	python phase2_compute/11_grouped_query_attention.py

run-sliding-window:  ## Run sliding window attention (Mistral-style)
	python phase2_compute/12_sliding_window_attention.py

run-attention-bias:  ## Run attention with bias (ALiBi)
	python phase2_compute/13_attention_bias.py

run-phase2: run-matmul-naive run-matmul-tiled run-matmul-autotuned \
           run-split-k run-fused-matmul run-transpose-variants \
           run-flash-v1 run-flash-v2 run-conv \
           run-flash-bwd run-gqa run-sliding-window run-attention-bias  ## Run all Phase 2 kernels

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

bench-phase1:  ## Phase 1 kernels: Triton vs Liger vs PyTorch (3-way comparison)
	python benchmarks/bench_phase1.py

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
