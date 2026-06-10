# Phase 1 — Triton 基础 (28 kernels)

## 学习路线

按 7 个 group 组织，由浅入深、同类相聚：

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

### Group 5: Reductions (12-20)
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

### Group 6: Normalizations (21-25)
| # | 文件 | Kernel | 关键概念 | 难度 |
|---|------|--------|----------|------|
| 21 | `21_layer_norm.py` | Layer Norm | 3-pass reduction, mean+var, affine | ⭐⭐⭐ |
| 22 | `22_rms_norm.py` | RMS Norm | 2-pass reduction, `tl.math.rsqrt` | ⭐⭐ |
| 23 | `23_group_norm.py` | Group Norm | 分组 reduction, spatial dims | ⭐⭐⭐ |
| 24 | `24_batch_norm.py` | BatchNorm1D | 跨 sample strided reduction | ⭐⭐⭐ |
| 25 | `25_residual_add_norm.py` | Residual+LayerNorm | Multi-input fusion, skip connection | ⭐⭐⭐ |

### Group 7: Position / Embedding / Optimizer (26-28)
| # | 文件 | Kernel | 关键概念 | 难度 |
|---|------|--------|----------|------|
| 26 | `26_rotary_embedding.py` | Rotary Embedding | Pairwise 2D rotation, RoPE | ⭐⭐ |
| 27 | `27_embedding.py` | Embedding Lookup | Gather/scatter, 随机访存 | ⭐⭐ |
| 28 | `28_adamw.py` | AdamW Optimizer | 多 buffer fusion, 6-in-1 kernel | ⭐⭐⭐ |

## 运行方式

```bash
# 单个文件
python phase1_fundamentals/01_vector_add.py
# ... (28 kernels total)

# 使用 Makefile
make run-vector-add   make run-sigmoid       make run-tanh
make run-leaky-relu   make run-relu-bias     make run-scale-bias-residual
make run-silu         make run-gelu          make run-dropout
make run-swiglu       make run-geglu
make run-vector-sum   make run-vector-max    make run-vector-norm  make run-welford
make run-logsumexp    make run-softmax       make run-cross-entropy make run-cumsum
make run-grad-clip
make run-layernorm    make run-rms-norm      make run-group-norm  make run-batch-norm
make run-residual-norm
make run-rope         make run-embedding     make run-adamw

# 全部运行
make run-phase1
```

## 性能对比

```bash
# 完整 3-way 对比 (Triton vs PyTorch vs Liger)
make bench-phase1

# 按 group 筛选
python benchmarks/bench_phase1.py --category elementwise    # Group 1-4, 7
python benchmarks/bench_phase1.py --category reduction      # Group 5
python benchmarks/bench_phase1.py --category normalization  # Group 6

# 快速模式
python benchmarks/bench_phase1.py --quick
```

## 学习建议

1. **Group 1** (01-03): 动手写第一个 kernel, 理解 GPU 执行模型
2. **Group 2** (04-06): 理解 operator fusion 为什么重要
3. **Group 3** (07-09): 学习复杂数学函数和随机数的 GPU 实现
4. **Group 4** (10-11): 掌握 gated activation 模式 (Llama FFN 的核心)
5. **Group 5** (12-20): reduction 全系列 — 从 sum/max 到 softmax/scan 到 grad clip
6. **Group 6** (21-25): 归一化全家桶 — 理解 LN/BN/GN/RMS 的差异
7. **Group 7** (26-28): 实战 — RoPE, Embedding, AdamW (LLM 训练全流程)
