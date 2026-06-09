# Triton Learn — 系统化 GPU 内核学习项目

以 **Triton** 为主线，兼顾 **Triton 编译器内部机制**，为未来扩展到 **CuTe/CUTLASS** 留好接口。

> 🎯 目标读者：有编译器后端经验的开发者（LLVM、寄存器分配、SSA IR），
> 想学习 GPU 编程和 Triton。

---

## 环境要求

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | ≥ 3.10 | |
| CUDA | ≥ 12.0 | Ampere (A100) 或 Hopper (H100) |
| Triton | ≥ 3.0 | `pip install triton` |
| PyTorch | ≥ 2.3 | 需要 CUDA 版本 |
| numpy | ≥ 1.24 | |
| matplotlib | ≥ 3.7 | Phase 3 layout 可视化（可选） |

## 快速上手

```bash
# 1. 安装依赖
pip install -e ".[dev]"

# 2. 验证环境
make check-env

# 3. 运行第一个 kernel
make run-vector-add

# 4. 运行测试
make test
```

---

## 学习路线图

```
Phase 1: Fundamentals          Phase 2: Compute             Phase 3: Compiler          Phase 4: CuTe
┌─────────────────┐           ┌─────────────────┐          ┌─────────────────┐        ┌─────────────────┐
│ 01_vector_add    │           │ 01_matmul_naive  │          │ 01_dump_ir       │        │ README.md       │
│ 02_fused_softmax │  ──────▶ │ 02_matmul_tiled  │ ──────▶ │ 02_layout_analysis│──────▶ │ placeholder.py  │
│ 03_fused_relu    │           │ 03_autotuned     │          │ 03_custom_pass    │        │ layout_algebra  │
│ 04_layer_norm    │           │ 04_flash_attn_v1 │          │ 04_ptx_analysis   │        │                 │
│                  │           │ 05_flash_attn_v2 │          │                   │        │                 │
└─────────────────┘           │ 06_depthwise_conv│          └─────────────────┘        └─────────────────┘
        ~1周                    └─────────────────┘                ~1-2周                     ~待扩展
                                        ~2-3周
```

### Phase 1 — Triton 基础（~1 周）

**目标**: 能用 Triton 写简单的 elementwise + reduction kernel

| 文件 | 关键概念 | 预计时间 |
|------|----------|----------|
| `01_vector_add.py` | `@triton.jit`, `tl.program_id`, `tl.load/store`, `@triton.autotune` | 30 min |
| `02_fused_softmax.py` | reduction (`tl.max`, `tl.sum`), masking, operator fusion | 45 min |
| `03_fused_relu_bias.py` | Fused elementwise ops, memory bandwidth analysis | 30 min |
| `04_layer_norm.py` | LayerNorm 数学, Welford 在线方差, shared memory reduction | 1 hr |

**关键概念清单**:
- [ ] Block/program 抽象 (vs CUDA thread)
- [ ] `tl.arange`, `tl.load`, `tl.store`, `tl.dot`
- [ ] Masking 和 boundary handling
- [ ] `tl.constexpr` 编译时常量
- [ ] Autotune (`@triton.autotune`, `triton.Config`)
- [ ] Reduction 操作 (`tl.sum`, `tl.max`)

**配套笔记**: `notes/00_gpu_execution_model.md`, `notes/01_triton_programming_model.md`

---

### Phase 2 — 核心计算内核（~2-3 周）

**目标**: 掌握 GEMM 优化 + Flash Attention

| 文件 | 关键概念 | 预计时间 |
|------|----------|----------|
| `01_matmul_naive.py` | Tiling (M/N/K), TFLOPS 计算 | 1 hr |
| `02_matmul_tiled.py` ⭐ | Shared memory, software pipeline, autotune, roofline | 3-4 hr |
| `03_matmul_autotuned.py` | 扩展搜索空间, GROUP_M swizzling | 2 hr |
| `04_flash_attention_v1.py` ⭐ | Online softmax, IO-aware algorithm | 3-4 hr |
| `05_flash_attention_v2.py` | Causal masking, 改进并行化 | 2 hr |
| `06_depthwise_conv.py` | 卷积映射到 GEMM | 2 hr |

**关键概念清单**:
- [ ] GEMM tiling: BLOCK_M × BLOCK_N × BLOCK_K
- [ ] Shared memory staging (double/triple buffering)
- [ ] `num_stages` / software pipelining
- [ ] `num_warps` / occupancy 权衡
- [ ] Online softmax: running max/sum rescaling
- [ ] IO-aware algorithm design (Flash Attention)
- [ ] Roofline analysis: compute bound vs memory bound

**配套笔记**: `notes/02_memory_hierarchy.md`

---

### Phase 3 — 编译器内部（~1-2 周）

**目标**: 理解 Triton 的编译管线，能读懂各阶段 IR

| 文件 | 关键概念 | 预计时间 |
|------|----------|----------|
| `01_dump_ir.py` | TTIR → TTGIR → LLVM IR → PTX | 1 hr |
| `02_layout_analysis.py` | BlockedEncoding, SliceEncoding, MmaEncoding | 2 hr |
| `03_custom_pass/` | MLIR pass 概念, Python AST 分析 | 2 hr |
| `04_ptx_analysis.py` | PTX 指令识别, 寄存器压力分析 | 1.5 hr |

**关键概念清单**:
- [ ] TTIR (tt dialect) vs TTGIR (ttg dialect)
- [ ] Layout encoding: `BlockedEncodingAttr`, `MmaEncodingAttr`
- [ ] `ConvertLayout` op → 可能导致 shared memory round-trip
- [ ] Key passes: `ConvertTritonToTritonGPU`, `TritonGPUPipeline`, `TritonGPUAccelerateMatmul`
- [ ] 寄存器压力 vs occupancy 的间接控制

**配套笔记**: `notes/03_triton_compiler_pipeline.md` ⭐ (最重要的一篇)

---

### Phase 4 — CuTe 预备（待扩展）

**目标**: 建立 CuTe 概念，为后续 C++ GPU 编程做准备

- [ ] 读 `notes/04_cute_preview.md`
- [ ] 运行 `phase4_cute/placeholder.py` 检查 CUTLASS 环境
- [ ] 理解 Layout = Shape × Stride 的代数

---

## Benchmarks — 你的 kernel vs 顶级算子库

```bash
make bench              # 快速对比（所有 kernel + liger-kernel + cuBLAS）
make bench-gemm         # 只看 GEMM
make bench-profile      # 带 torch.profiler + chrome traces

# 三层独立对比（新）
make check-gpu          # 打印 GPU 硬件规格 + roofline
make bench-matmul       # GEMM: 你的 Triton vs cuBLAS vs roofline ceiling
make bench-attn         # Attention: Flash Attn vs SDPA vs naive
make bench-elem         # Elementwise/norm: Triton vs Liger vs PyTorch
make bench-all          # 全部三层 benchmark + 保存 JSON 结果
```

在 `benchmarks/` 目录下有一套完整的性能对比系统，自动加载你的 Triton kernel，
对比 **PyTorch (cuBLAS/cuDNN)**、**Liger Kernel**、**flash-attn**（Tri Dao 官方实现），
以及 **硬件 roofline ceiling**。

详见 `benchmarks/README.md` 和 `notes/05_benchmarking_methodology.md`。

---

## 项目结构

```
triton-learn/
├── README.md                    # 本文件
├── pyproject.toml               # 依赖管理
├── Makefile                     # 常用命令
│
├── notes/                       # 学习笔记（Markdown）
│   ├── 00_gpu_execution_model.md
│   ├── 01_triton_programming_model.md
│   ├── 02_memory_hierarchy.md
│   ├── 03_triton_compiler_pipeline.md  ← 最重要
│   ├── 04_cute_preview.md
│   └── 05_benchmarking_methodology.md  ← 如何正确做 GPU benchmark
│
├── phase1_fundamentals/         # 第一阶段：Triton 基础
│   ├── 01_vector_add.py
│   ├── 02_fused_softmax.py
│   ├── 03_fused_relu_bias.py
│   ├── 04_layer_norm.py
│   └── benchmarks/
│
├── phase2_compute/              # 第二阶段：核心计算内核
│   ├── 01_matmul_naive.py
│   ├── 02_matmul_tiled.py       ← 重点：生产级 GEMM
│   ├── 03_matmul_autotuned.py   ← 实现完成
│   ├── 04_flash_attention_v1.py ← 重点：Flash Attention
│   ├── 05_flash_attention_v2.py ← 实现完成（causal mask）
│   ├── 06_depthwise_conv.py     ← 实现完成
│   └── benchmarks/
│
├── phase3_compiler/             # 第三阶段：编译器内部
│   ├── 01_dump_ir.py
│   ├── 02_layout_analysis.py
│   ├── 03_custom_pass/
│   └── 04_ptx_analysis.py
│
├── phase4_cute/                 # 第四阶段：CuTe 预备
│   ├── README.md
│   ├── placeholder.py
│   └── 00_layout_algebra.md
│
├── utils/                       # 工具函数
│   ├── profiler.py              # GPU kernel 性能测量 + bench_compare()
│   ├── checker.py               # 数值正确性验证
│   ├── ir_dump.py               # IR dump 工具
│   └── roofline.py              # GPU 硬件规格 + roofline 模型分析
│
├── tests/                       # 测试
│   ├── conftest.py
│   ├── test_phase1.py
│   └── test_phase2.py
│
├── benchmarks/                  # 独立 benchmark 套件
│   ├── README.md
│   ├── hardware_spec.py         # GPU 硬件规格自动检测
│   ├── bench_runner.py          # 统一 runner（兼容旧接口）
│   ├── bench_cases.py           # benchmark case 定义
│   ├── bench_matmul.py          # GEMM 三层对比（独立运行）
│   ├── bench_attention.py       # Attention 三层对比（独立运行）
│   ├── bench_elementwise.py     # Elementwise/norm 对比（独立运行）
│   ├── references/              # SotA 参照实现封装
│   │   ├── cublas_gemm.py       # cuBLAS（通过 torch.mm）
│   │   ├── flash_attn_ref.py    # flash-attn / torch SDPA
│   │   └── liger_ref.py         # Liger Kernel 封装
│   └── results/                 # benchmark 结果存储（JSON）
└── traces/                      # torch.profiler chrome traces
```

---

## 常用命令

```bash
# 环境
make check-env         # 验证环境

# 运行 kernel
make run-vector-add    # Phase 1
make run-softmax
make run-matmul-tiled  # Phase 2
make run-flash-v1

# 编译器
make dump-ir           # Phase 3: dump 各阶段 IR
make layout-analysis
make ptx-analysis

# 测试
make test              # CPU-only tests
make test-gpu          # GPU tests (needs CUDA)
make test-all          # All tests

# Profiling (requires NVIDIA Nsight Compute)
make profile-matmul    # ncu --set full

# 清理
make clean
```

---

## 代码风格

所有 Triton kernel 遵循统一的注释风格：

```python
@triton.jit
def example_kernel(
    ...,
    BLOCK_SIZE: tl.constexpr,  # [COMPILER] 编译时常量，类比模板参数
):
    """
    Kernel 功能描述。
    数学公式（如适用）。
    """
    pid = tl.program_id(axis=0)  # blockIdx.x

    # 加载数据 — 编译器处理 coalescing
    x = tl.load(ptr + offsets, mask=mask, other=0.0)

    # 计算
    result = ...

    # 写回 — 编译器处理 alignment
    tl.store(out_ptr + offsets, result, mask=mask)
```

注释规范：
- **GPU 语义注释**: 解释 GPU 执行层面发生的事情
- **`# [COMPILER]`**: 标注与编译器内部行为相关的注释
- **数学公式**: 在涉及算法的 kernel 中用注释写出公式
- **PERFORMANCE NOTES**: 每个文件末尾的性能分析段落

---

## 参考资料

### Triton
- [Triton 论文 (Tillet et al., 2019)](https://dl.acm.org/doi/10.1145/3315508.3329973) — 原始 Triton 论文
- [Triton 论文 v2 (Tillet et al., 2023)](https://www.eecs.harvard.edu/~htk/publication/2023-pact-tillet-kung-cox.pdf) — 编译器细节
- [Triton 官方文档](https://triton-lang.org/)
- [Triton 官方 Tutorials](https://triton-lang.org/main/getting-started/tutorials/)

### Flash Attention
- [FlashAttention (Dao et al., NeurIPS 2022)](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2 (Dao, 2023)](https://arxiv.org/abs/2307.08691)
- [FlashAttention-3 (Shah et al., 2024)](https://arxiv.org/abs/2407.08608)

### GPU 架构
- [NVIDIA H100 Whitepaper](https://resources.nvidia.com/en-us-tensor-core)
- [NVIDIA A100 Whitepaper](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

### CUTLASS / CuTe
- [CUTLASS GitHub](https://github.com/NVIDIA/cutlass)
- [CuTe Quick Start](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/00_quickstart.md)
- [CUTLASS 3.x Design](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/programming_guidelines.md)

### MLIR
- [MLIR Documentation](https://mlir.llvm.org/docs/)
- [MLIR Tutorial](https://mlir.llvm.org/docs/Tutorials/)

---

## License

MIT
