# Phase 1 — Triton 基础 (50 kernels)

## 学习路线

按 10 个 group 组织，由浅入深、同类相聚：

### Group 1: Hello World + 基础激活 (01-03)
| # | 文件 | Kernel | 关键概念 | 难度 |
|---|------|--------|----------|------|
| 1 | `01_vector_add.py` | Vector Add | `@triton.jit`, `tl.program_id`, `tl.load/store`, autotune | ⭐ |
| 2 | `02_sigmoid.py` | Sigmoid | `tl.sigmoid`, MUFU 硬件加速 | ⭐ |
| 3 | `03_tanh.py` | Tanh | 手写数学函数, sigmoid→tanh 等价变换 | ⭐ |

### Group 2: Elementwise Fusion (04-06)
| # | 文件 | Kernel | 关键概念 | 难度 |
|---|------|--------|----------|------|
| 4 | `04_leaky_relu.py` | LeakyReLU / PReLU | `tl.where` 分支消除, 参数化激活 | ⭐ |
| 5 | `05_fused_relu_bias.py` | Fused ReLU+Bias | 2-op fusion, bandwidth analysis | ⭐ |
| 6 | `06_fused_scale_bias_residual.py` | Scale+Bias+Residual | 3-input fusion, ResNet pattern | ⭐⭐ |

### Group 3: 高级激活 + Dropout (07-09)
| # | 文件 | Kernel | 关键概念 | 难度 |
|---|------|--------|----------|------|
| 7 | `07_silu.py` | SiLU / Swish | `x*sigmoid(x)`, Llama 激活 | ⭐ |
| 8 | `08_gelu.py` | GELU | tanh 近似, BERT/GPT 激活 | ⭐⭐ |
| 9 | `09_dropout.py` | Dropout | Philox RNG (`tl.rand`), inverted dropout | ⭐⭐ |

### Group 4: Gated Activations (10-11)
| # | 文件 | Kernel | 关键概念 | 难度 |
|---|------|--------|----------|------|
| 10 | `10_swiglu.py` | SwiGLU | Fused `gate*SiLU(up)`, Llama FFN | ⭐⭐ |
| 11 | `11_geglu.py` | GeGLU | Fused `gate*GELU(up)`, 对比 SwiGLU | ⭐⭐ |

### Group 5: Reductions (12-20, 29-31)
| # | 文件 | Kernel | 关键概念 | 难度 |
|---|------|--------|----------|------|
| 12 | `12_vector_sum.py` | Vector Sum | `tl.sum`, block 内 reduction, atomic_add | ⭐ |
| 13 | `13_vector_max.py` | Vector Max | `tl.max`, atomic_max, identity element | ⭐ |
| 14 | `14_vector_norm_l2.py` | L2 Vector Norm | sum(x²)+sqrt, compute+reduce 组合 | ⭐ |
| 15 | `15_welford_mean_var.py` | Welford Mean+Var | 1-pass online 算法, 比 2-pass 少读 50% | ⭐⭐⭐ |
| 16 | `16_logsumexp.py` | Row-Wise LogSumExp | max-subtraction trick, softmax 对数版 | ⭐⭐ |
| 17 | `17_fused_softmax.py` | Fused Softmax | max+sum reduction, online softmax | ⭐⭐ |
| 18 | `18_cross_entropy.py` | Cross Entropy Loss | log_softmax, max-subtraction trick | ⭐⭐⭐ |
| 19 | `19_cumsum.py` | Cumsum / Prefix Scan | Block-level scan, cross-block carry | ⭐⭐⭐ |
| 20 | `20_gradient_clipping.py` | Gradient Clipping | `tl.atomic_add`, 全局 norm reduction | ⭐⭐ |
| 29 | `29_parallel_mean_var.py` | Parallel Mean+Var | E[X²]-(E[X])², tl.sum 并行归约 vs Welford 串行 | ⭐⭐ |
| 30 | `30_argmax_reduce.py` | Argmax Reduce | 携带 index 的 reduction, argmax pattern | ⭐⭐ |
| 31 | `31_topk_selection.py` | Top-K Selection | 阈值过滤 compaction, 比全排序更高效 | ⭐⭐⭐ |

### Group 6: Loss Functions (32-34)
| # | 文件 | Kernel | 关键概念 | 难度 |
|---|------|--------|----------|------|
| 32 | `32_mse_loss.py` | Fused MSE Loss | elementwise diff² + reduction 融合 | ⭐⭐ |
| 33 | `33_hinge_loss.py` | Fused Hinge Loss | max(0, margin-y*pred), 无分支比较 | ⭐⭐ |
| 34 | `34_l1_loss.py` | Fused L1 Loss (MAE) | tl.abs (符号位清除), L1 vs MSE 鲁棒性 | ⭐⭐ |

### Group 7: MobileNet 激活三件套 (35-37)
| # | 文件 | Kernel | 关键概念 | 难度 |
|---|------|--------|----------|------|
| 35 | `35_relu6_clamp.py` | ReLU6 Clamp | min/max sandwich, int8 量化友好 | ⭐ |
| 36 | `36_hard_sigmoid.py` | Hard Sigmoid | 分段线性近似, 省 exp → 推理加速 | ⭐ |
| 37 | `37_hard_swish.py` | Hard Swish | x*hard_sigmoid(x), MobileNetV3 核心 | ⭐⭐ |

### Group 8: BLAS Primitives & Data Movement (38-40)
| # | 文件 | Kernel | 关键概念 | 难度 |
|---|------|--------|----------|------|
| 38 | `38_vector_dot.py` | Vector Dot Product | 内积 = elementwise× + reduction, 通向 matmul | ⭐⭐ |
| 39 | `39_transpose_2d.py` | 2D Transpose | Coalesced vs strided access, shared memory 中转 | ⭐⭐ |
| 40 | `40_concat.py` | Concatenation | 多 base pointer 访存, tl.where 条件选择 | ⭐ |

### Group 9: Pooling & Attention Blocks (41-45)
| # | 文件 | Kernel | 关键概念 | 难度 |
|---|------|--------|----------|------|
| 41 | `41_max_pool1d.py` | Max Pool 1D | Sliding window + max reduction | ⭐⭐ |
| 42 | `42_avg_pool1d.py` | Avg Pool 1D | Sliding window + mean reduction | ⭐⭐ |
| 43 | `43_scaled_dot_product.py` | Scaled Dot-Product | QK^T/sqrt(d) — Attention building block | ⭐⭐⭐ |
| 44 | `44_causal_mask.py` | Causal Mask | 下三角 mask 生成 + softmax additive mask | ⭐⭐ |
| 45 | `45_one_hot.py` | One-Hot Encoding | Scatter store 模式 vs gather | ⭐⭐ |

### Group 10: Optimizer & Similarity Patterns (46-50)
| # | 文件 | Kernel | 关键概念 | 难度 |
|---|------|--------|----------|------|
| 46 | `46_weight_decay.py` | Weight Decay | In-place 参数更新, AdamW 第一步 | ⭐ |
| 47 | `47_ema.py` | Exponential Moving Avg | EMA = β*old+(1-β)*new, BatchNorm running stats | ⭐⭐ |
| 48 | `48_cosine_similarity.py` | Cosine Similarity | dot/(|x||y|), 多统计量单 pass 融合 | ⭐⭐ |
| 49 | `49_gelu_accurate.py` | Exact GELU vs Tanh Approx | tl.math.erf 用法, 精度 vs 速度 trade-off | ⭐⭐⭐ |
| 50 | `50_fused_bias_gelu.py` | Fused Bias+GELU | 3-op fusion (load→add→activate→store), 省 50% HBM | ⭐⭐ |

## 运行方式

```bash
# 单个文件
python phase1_fundamentals/01_vector_add.py
# ... (50 kernels total)

# 使用 Makefile
# Group 1-4
make run-vector-add   make run-sigmoid       make run-tanh
make run-leaky-relu   make run-relu-bias     make run-scale-bias-residual
make run-silu         make run-gelu          make run-dropout
make run-swiglu       make run-geglu

# Group 5 (Reductions)
make run-vector-sum   make run-vector-max    make run-vector-norm
make run-welford      make run-logsumexp     make run-softmax
make run-cross-entropy make run-cumsum       make run-grad-clip
make run-mean-var     make run-argmax        make run-topk

# Group 6 (Loss Functions)
make run-mse          make run-hinge         make run-l1

# Group 7 (MobileNet Activations)
make run-relu6        make run-hard-sigmoid  make run-hard-swish

# Group 8 (BLAS + Data Movement)
make run-dot          make run-transpose     make run-concat

# Group 9 (Pooling + Attention)
make run-max-pool     make run-avg-pool      make run-scaled-dot
make run-causal-mask  make run-one-hot

# Group 10 (Optimizer + Similarity)
make run-weight-decay make run-ema           make run-cosine-sim
make run-gelu-exact   make run-fused-bias-gelu

# 全部运行
make run-phase1
```

## 性能对比

```bash
# 完整 3-way 对比 (Triton vs PyTorch vs Liger)
make bench-phase1

# 按 group 筛选
python benchmarks/bench_phase1.py --category elementwise    # Group 1-4, 7-8
python benchmarks/bench_phase1.py --category reduction      # Group 5
python benchmarks/bench_phase1.py --category normalization  # Group 6 (old)
python benchmarks/bench_phase1.py --category loss           # Group 6 (new)
python benchmarks/bench_phase1.py --category pooling        # Group 9

# 快速模式
python benchmarks/bench_phase1.py --quick
```

## 学习建议

1. **Group 1** (01-03): 动手写第一个 kernel, 理解 GPU 执行模型
2. **Group 2** (04-06): 理解 operator fusion 为什么重要
3. **Group 3** (07-09): 学习复杂数学函数和随机数的 GPU 实现
4. **Group 4** (10-11): 掌握 gated activation 模式 (Llama FFN 的核心)
5. **Group 5** (12-20, 29-31): reduction 全系列 — 从 sum/max 到 softmax/scan 到 topk
6. **Group 6** (32-34): loss function 融合 — MSE/L1/Hinge
7. **Group 7** (35-37): MobileNet 高效推理激活 — piecewise linear 近似
8. **Group 8** (38-40): BLAS 基础 (dot product) + 数据搬运 (transpose/concat)
9. **Group 9** (41-45): Pooling + Attention building blocks
10. **Group 10** (46-50): 优化器 (weight decay/EMA) + GELU 精确 vs 近似
