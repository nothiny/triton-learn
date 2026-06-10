# Triton Kernels — GPU 算子学习项目

以 **Triton** (OpenAI) 为主线，对比 **cuTile Python** (NVIDIA)，兼顾 Triton 编译器内部机制。

> 🎯 目标读者：有编译器后端经验的开发者（LLVM、寄存器分配、SSA IR），想系统学习 GPU 算子编写。

---

## 环境要求

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | ≥ 3.10 | |
| CUDA | ≥ 12.0 | Triton 需要 Ampere+；cuTile 需要 13.1+ (Blackwell/Ampere/Ada) |
| Triton | 3.6.0 | `pip install triton` |
| PyTorch | ≥ 2.3 | CUDA 版本 |
| cuTile | 可选 | `pip install cuda-tile[tileiras]` (Phase 4) |

## 快速上手

```bash
pip install -e ".[dev]"
make check-env
make run-vector-add     # 第一个 kernel
make bench-phase1       # 全部 23 个 kernel 的性能对比
make test               # CPU 测试
```

---

## 学习路线图

```
Phase 1: Fundamentals (23 kernels)    Phase 2: Compute             Phase 3: Compiler          Phase 4: cuTile
┌──────────────────────────┐         ┌─────────────────┐          ┌─────────────────┐        ┌─────────────────┐
│ Group 1: Basics (01-03)   │         │ 01_matmul_naive  │          │ 01_dump_ir       │        │ cuTile vs Triton │
│ Group 2: Fusion (04-06)   │ ──────▶ │ 02_matmul_tiled  │ ──────▶ │ 02_layout_analysis│──────▶ │ 对比分析         │
│ Group 3: Activations (07-09)│       │ 03_autotuned     │          │ 03_custom_pass    │        │ Tile IR 理解     │
│ Group 4: Gated (10-11)    │         │ 04_flash_attn_v1 │          │ 04_ptx_analysis   │        │                 │
│ Group 5: Reductions (12-15)│        │ 05_flash_attn_v2 │          │                   │        │                 │
│ Group 6: Normalizations    │        │ 06_depthwise_conv│          └─────────────────┘        └─────────────────┘
│         (16-20)            │        └─────────────────┘                ~1-2 周                   ~待扩展
│ Group 7: Embed/Optim (21-23)│              ~2-3 周
└──────────────────────────┘
          ~2-3 周
```

---

## Phase 1 — Triton 基础 (23 kernels, 7 个学习组)

**目标**: 用 Triton 写出 production-quality 的 elementwise + reduction + normalization kernel。

### Group 1: Hello World + 基础激活 (01-03)

| # | 文件 | Kernel | 关键概念 | 难度 |
|---|------|--------|----------|:--:|
| 1 | `01_vector_add.py` | Vector Add | `@triton.jit`, `tl.program_id`, `tl.load/store`, autotune | ⭐ |
| 2 | `02_sigmoid.py` | Sigmoid | `tl.sigmoid`, MUFU 硬件加速 | ⭐ |
| 3 | `03_tanh.py` | Tanh | sigmoid→tanh 等价变换: `2σ(2x)-1` | ⭐ |

### Group 2: Elementwise Fusion (04-06)

| # | 文件 | Kernel | 关键概念 | 难度 |
|---|------|--------|----------|:--:|
| 4 | `04_leaky_relu.py` | LeakyReLU / PReLU | `tl.where` 分支消除, 可学习参数 | ⭐ |
| 5 | `05_fused_relu_bias.py` | Fused ReLU+Bias | 2-op fusion, bandwidth analysis | ⭐ |
| 6 | `06_fused_scale_bias_residual.py` | Scale+Bias+Residual | 3-input fusion, ResNet pattern | ⭐⭐ |

### Group 3: 高级激活 + Dropout (07-09)

| # | 文件 | Kernel | 关键概念 | 难度 |
|---|------|--------|----------|:--:|
| 7 | `07_silu.py` | SiLU / Swish | `x·σ(x)`, Llama 激活函数 | ⭐ |
| 8 | `08_gelu.py` | GELU | tanh 近似, BERT/GPT 激活 | ⭐⭐ |
| 9 | `09_dropout.py` | Dropout | Philox RNG (`tl.rand`), inverted dropout | ⭐⭐ |

### Group 4: Gated Activations (10-11)

| # | 文件 | Kernel | 关键概念 | 难度 |
|---|------|--------|----------|:--:|
| 10 | `10_swiglu.py` | SwiGLU | Fused `gate·SiLU(up)`, Llama FFN | ⭐⭐ |
| 11 | `11_geglu.py` | GeGLU | Fused `gate·GELU(up)`, 对比 SwiGLU | ⭐⭐ |

### Group 5: Reductions (12-15)

| # | 文件 | Kernel | 关键概念 | 难度 |
|---|------|--------|----------|:--:|
| 12 | `12_fused_softmax.py` | Fused Softmax | max+sum reduction, online algorithm | ⭐⭐ |
| 13 | `13_cross_entropy.py` | Cross Entropy Loss | log_softmax, max-subtraction trick | ⭐⭐⭐ |
| 14 | `14_cumsum.py` | Cumsum / Prefix Scan | Block-level scan, cross-block carry | ⭐⭐⭐ |
| 15 | `15_gradient_clipping.py` | Gradient Clipping | `tl.atomic_add`, 全局 norm reduction | ⭐⭐ |

### Group 6: Normalizations (16-20)

| # | 文件 | Kernel | 关键概念 | 难度 |
|---|------|--------|----------|:--:|
| 16 | `16_layer_norm.py` | Layer Norm | 3-pass: mean→var→norm+affine | ⭐⭐⭐ |
| 17 | `17_rms_norm.py` | RMS Norm | 2-pass, `tl.math.rsqrt`, Llama norm | ⭐⭐ |
| 18 | `18_group_norm.py` | Group Norm | 分组 reduction, G=1→LN, G=C→IN | ⭐⭐⭐ |
| 19 | `19_batch_norm.py` | BatchNorm1D | 跨 sample strided reduction | ⭐⭐⭐ |
| 20 | `20_residual_add_norm.py` | Residual+LayerNorm | Transformer skip connection fusion | ⭐⭐⭐ |

### Group 7: Position / Embedding / Optimizer (21-23)

| # | 文件 | Kernel | 关键概念 | 难度 |
|---|------|--------|----------|:--:|
| 21 | `21_rotary_embedding.py` | Rotary Embedding | Pairwise 2D rotation, RoPE | ⭐⭐ |
| 22 | `22_embedding.py` | Embedding Lookup | Gather/scatter, 随机访存模式 | ⭐⭐ |
| 23 | `23_adamw.py` | AdamW Optimizer | 6-in-1 fusion, momentum buffer update | ⭐⭐⭐ |

---

## Phase 2 — 核心计算内核（~2-3 周）

**目标**: 掌握 GEMM 优化 + Flash Attention

| 文件 | 关键概念 | 预计 |
|------|----------|:--:|
| `01_matmul_naive.py` | Tiling (M/N/K), TFLOPS 计算 | 1h |
| `02_matmul_tiled.py` ⭐ | Shared memory, software pipeline, autotune, roofline | 3-4h |
| `03_matmul_autotuned.py` | GROUP_M swizzling, 扩展搜索空间 | 2h |
| `04_flash_attention_v1.py` ⭐ | Online softmax, IO-aware 算法 | 3-4h |
| `05_flash_attention_v2.py` | Causal masking, 改进并行化 | 2h |
| `06_depthwise_conv.py` | 卷积→GEMM 映射 | 2h |

**关键概念**: GEMM tiling, shared memory staging, software pipelining, `num_stages`/`num_warps` 权衡, online softmax, IO-aware design, roofline analysis

**配套笔记**: `notes/02_memory_hierarchy.md`, `notes/08_gemm_advanced.md`

---

## Phase 3 — 编译器内部（~1-2 周）

**目标**: 理解 Triton 编译管线，能读懂各阶段 IR

| 文件 | 关键概念 | 预计 |
|------|----------|:--:|
| `01_dump_ir.py` | TTIR → TTGIR → LLVM IR → PTX | 1h |
| `02_layout_analysis.py` | BlockedEncoding, SliceEncoding, MmaEncoding | 2h |
| `03_custom_pass/` | MLIR pass 概念, Python AST 分析 | 2h |
| `04_ptx_analysis.py` | PTX 指令识别, 寄存器压力分析 | 1.5h |

**关键概念**: TTIR vs TTGIR, layout encoding, `ConvertLayout` op, key passes, 寄存器压力 vs occupancy

**配套笔记**: `notes/03_triton_compiler_pipeline.md` ⭐

---

## Phase 4 — cuTile Python: NVIDIA 的 Triton 替代（待扩展）

**目标**: 学习 NVIDIA cuTile Python，对比 Triton 的设计哲学

cuTile Python (`cuda-tile` on PyPI) 是 NVIDIA 官方的 GPU kernel 编程语言，基于 Tile IR。

| 维度 | Triton (OpenAI) | cuTile (NVIDIA) |
|------|:---:|:---:|
| 编译器 IR | Triton IR → TTGIR → LLVM | Tile IR → PTX |
| 抽象层级 | Block-level (program) | Tile-level |
| 硬件支持 | Ampere+ | Blackwell / Ampere / Ada |
| CUDA 要求 | ≥ 12.0 | ≥ 13.1 |
| 生态 | 独立开源 | NVIDIA 官方, CUTLASS 生态 |
| 学习曲线 | 中等 | 较陡（底层控制更多） |

**学习目标**:
- [ ] 安装 cuTile: `pip install cuda-tile[tileiras]`
- [ ] 用 cuTile 重写 Phase 1 的 vector_add 作为对比
- [ ] 理解 Tile IR 和 Triton IR 的设计差异
- [ ] 对比 Triton vs cuTile 在同 kernel 上的性能和易用性

**配套文件**: `phase4_cutile/`, `notes/04_cutile_preview.md`

---

## Benchmarks — 你的 kernel vs 顶级算子库

```bash
make bench-phase1             # Phase 1: 23 kernels (Triton vs PyTorch vs Liger)
make bench-matmul             # GEMM: Triton vs cuBLAS vs roofline
make bench-attn               # Attention: Flash Attn vs SDPA vs naive
make bench-elem               # Elementwise/norm: Triton vs Liger vs PyTorch
make bench-all                # 全部 + 保存 JSON
make check-gpu                # GPU 硬件规格 + roofline ridge points
```

详见 `benchmarks/README.md`。

---

## 项目结构

```
triton-kernels/
├── README.md
├── pyproject.toml
├── Makefile
│
├── notes/                       # 学习笔记
│   ├── 00_gpu_execution_model.md
│   ├── 01_triton_programming_model.md
│   ├── 02_memory_hierarchy.md
│   ├── 03_triton_compiler_pipeline.md  ← 最重要
│   ├── 04_cutile_preview.md
│   ├── 05_benchmarking_methodology.md
│   ├── 08_gemm_advanced.md
│   ├── 10_hopper_architecture.md
│   └── 12_optimization_cases.md
│
├── phase1_fundamentals/         # 23 Triton kernels, 7 groups
│   ├── 01_vector_add.py         ← 入门起点
│   ├── 02_sigmoid.py  ...  23_adamw.py
│   └── README.md
│
├── phase2_compute/              # GEMM + Flash Attention
│   ├── 01_matmul_naive.py
│   ├── 02_matmul_tiled.py       ← 生产级 GEMM
│   ├── 03_matmul_autotuned.py
│   ├── 04_flash_attention_v1.py ← Flash Attention
│   ├── 05_flash_attention_v2.py
│   └── 06_depthwise_conv.py
│
├── phase3_compiler/             # Triton 编译器内部
│   ├── 01_dump_ir.py
│   ├── 02_layout_analysis.py
│   ├── 03_custom_pass/
│   └── 04_ptx_analysis.py
│
├── phase4_cutile/               # cuTile Python 探索
│   ├── README.md
│   ├── placeholder.py
│   └── 00_layout_algebra.md
│
├── utils/                       # 工具
│   ├── profiler.py              # Kernel 性能测量 + bench_compare()
│   ├── checker.py               # 数值正确性验证
│   ├── ir_dump.py               # IR dump 工具
│   └── roofline.py              # GPU 硬件规格 + roofline 分析
│
├── tests/
│   ├── conftest.py
│   ├── test_phase1.py
│   └── test_phase2.py
│
└── benchmarks/                  # Benchmark 套件
    ├── bench_phase1.py          # Phase 1 23 kernel 三向对比
    ├── bench_matmul.py          # GEMM standalone
    ├── bench_attention.py       # Attention standalone
    ├── bench_elementwise.py     # Elementwise/norm standalone
    ├── bench_runner.py          # 统一 runner
    ├── bench_cases.py           # 用例定义
    ├── hardware_spec.py         # GPU 规格检测
    └── references/
        ├── cublas_gemm.py
        ├── flash_attn_ref.py
        └── liger_ref.py
```

---

## 常用命令

```bash
# 环境
make check-env

# Phase 1 — 按 group
make run-vector-add    make run-sigmoid      make run-tanh          # Group 1
make run-leaky-relu    make run-relu-bias    make run-scale-bias-residual  # Group 2
make run-silu          make run-gelu         make run-dropout       # Group 3
make run-swiglu        make run-geglu                              # Group 4
make run-softmax       make run-cross-entropy make run-cumsum make run-grad-clip  # Group 5
make run-layernorm     make run-rms-norm     make run-group-norm make run-batch-norm make run-residual-norm  # Group 6
make run-rope          make run-embedding    make run-adamw         # Group 7

# Phase 2
make run-matmul-tiled  make run-flash-v1

# Benchmark
make bench-phase1      make bench-matmul     make bench-all

# 编译器
make dump-ir           make layout-analysis  make ptx-analysis

# 测试
make test              make test-gpu         make test-all

# Profiling + 清理
make profile-matmul    make clean
```

---

## 代码风格

每个 kernel 文件遵循统一结构：

```python
# 1. 文件头 docstring: 公式、学习目标
# 2. @triton.autotune (可选)
# 3. @triton.jit kernel → tl.load → compute → tl.store
# 4. Python 包装函数
# 5. main(): 正确性 → 性能对比 (Triton vs PyTorch vs Liger)
# 6. PERFORMANCE NOTES: roofline, bottleneck, 优化方向
```

注释规范：
- `# [COMPILER]` — 编译器相关行为
- GPU 语义注释 — 解释 coalescing, banking, warp scheduling
- 数学公式 — 在算法 kernel 中写出推导

---

## 参考资料

### Triton
- [Triton 论文 (Tillet et al., 2019)](https://dl.acm.org/doi/10.1145/3315508.3329973)
- [Triton 官方文档](https://triton-lang.org/)
- [Triton 官方 Tutorials](https://triton-lang.org/main/getting-started/tutorials/)

### Flash Attention
- [FlashAttention (Dao et al., NeurIPS 2022)](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2 (Dao, 2023)](https://arxiv.org/abs/2307.08691)

### cuTile Python (NVIDIA)
- [cuTile Python 官方文档](https://docs.nvidia.com/cuda/cutile-python)
- [GitHub: NVIDIA/cutile-python](https://github.com/NVIDIA/cutile-python)
- [TileGym 示例](https://github.com/NVIDIA/TileGym)

### GPU 架构
- [NVIDIA H100 Whitepaper](https://resources.nvidia.com/en-us-tensor-core)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

### CUTLASS
- [CUTLASS GitHub](https://github.com/NVIDIA/cutlass)

### MLIR
- [MLIR Documentation](https://mlir.llvm.org/docs/)

---

## License

MIT
