# Notes — 学习笔记

## 阅读顺序

### 基础篇（先读这些）

| # | 文件 | 内容 | 预计时间 | 前置知识 |
|---|------|------|----------|----------|
| 00 | `00_gpu_execution_model.md` | GPU 执行模型：SM/warp/内存层级/Tensor Core | 45 min | 无，零基础可读 |
| 01 | `01_triton_programming_model.md` | Triton 编程模型：program 抽象、核心 API | 30 min | 00 |
| 02 | `02_memory_hierarchy.md` | 内存层级：coalescing/bank conflict/software pipeline | 30 min | 00, 01 |

### 数值与硬件基础

| # | 文件 | 内容 | 预计时间 | 前置知识 |
|---|------|------|----------|----------|
| 03 | `03_numerical_precision.md` | 数值精度：fp32/fp16/bf16/tf32/fp8 选择指南 | 30 min | 01 |
| 04 | `04_tensor_core_deep.md` | Tensor Core 深入：MMA 指令、FP8、sparsity | 30 min | 00 |

### 核心实践

| # | 文件 | 内容 | 预计时间 | 前置知识 |
|---|------|------|----------|----------|
| 05 | `05_cuda_to_triton.md` | CUDA → Triton 迁移指南：概念映射、代码翻译 | 20 min | 01 + CUDA 经验 |
| 06 | `06_kernel_patterns.md` | 常见 Kernel 模式：reduce/scan/gather/conv | 30 min | 01, 02 |
| 07 | `07_triton_autograd.md` | Triton 反向传播：写 Backward Pass + Autograd 集成 | 45 min | 01 + PyTorch 基础 |
| 08 | `08_debugging_triton.md` | 🐛 调试 Triton Kernel：工具、常见 Bug、工作流 | 30 min | 01 |

### 性能优化方法论

| # | 文件 | 内容 | 预计时间 | 前置知识 |
|---|------|------|----------|----------|
| 09 | `09_benchmarking_methodology.md` | Benchmark 方法论：正确测量 GPU 性能 | 30 min | 01 |
| 10 | `10_autotuning_strategy.md` | Autotuning 策略：搜索空间设计、Prune、Key 选择 | 30 min | 01 |
| 11 | `11_gemm_advanced.md` | GEMM 优化进阶：warp tiling/pipeline/swizzling | 45 min | 00-02 |
| 12 | `12_optimization_cases.md` | 真实优化案例：GEMM/Softmax/LN/RMSNorm/SwiGLU | 45 min | 02, 11 |
| 13 | `13_production_checklist.md` | 生产级 Triton Kernel 优化清单 | 30 min | 01, 02, 15 |

### 高级架构与调度

| # | 文件 | 内容 | 预计时间 | 前置知识 |
|---|------|------|----------|----------|
| 14 | `14_hopper_architecture.md` | Hopper (H100) 架构：wgmma/TMA/FP8/cluster | 30 min | 00, 04 |
| 15 | `15_block_pointer_api.md` | ⭐ Block Pointer API 完全指南：`make_block_ptr`、`advance`、`boundary_check`、`order` | 40 min | 01, 02 + 写过 tiled GEMM |
| 16 | `16_tma_async_copy.md` | TMA 与异步数据搬运：`cp.async`→TMA 演进、Hopper 硬件、warp specialization | 35 min | 00, 02, 15, 14 |
| 17 | `17_persistent_kernels.md` | Persistent Kernel 与 Stream-K：atomic work dispatch、动态 K 分解 | 30 min | 01 |
| 18 | `18_multi_gpu.md` | Multi-GPU：NVLink/all-reduce/tensor parallelism | 20 min | 01 |

### 算法深度剖析

| # | 文件 | 内容 | 预计时间 | 前置知识 |
|---|------|------|----------|----------|
| 19 | `19_flash_attention_math.md` | ⭐ Flash Attention 完整数学推导 | 45 min | 01 |

### 编译器深度（Phase 4 配套）

| # | 文件 | 内容 | 预计时间 | 前置知识 |
|---|------|------|----------|----------|
| 20 | `20_triton_internals.md` | Triton 内部：JIT/Cache/driver API | 20 min | 21 |
| 21 | `21_triton_compiler_pipeline.md` | ⭐ 编译器管线：TTIR→TTGIR→LLVM→PTX | 1 hr | 01 + MLIR 基础 |
| 22 | `22_phase4_compiler_internals.md` | ⭐ Triton 编译器内部：AST→TTIR、Lowering 追踪、Pipelining、寄存器分配、Autotuner、PTX→SASS | 1 hr | 21, 20 |
| 23 | `23_phase4_mlir_framework.md` | ⭐ MLIR 框架深度：核心抽象、`tt`/`ttg` Dialect 设计、Pass 系统、Pattern Rewriting | 45 min | 21, 22 |
| 24 | `24_phase4_layout_encoding.md` | ⭐ Layout Encoding 完全指南：5 种 Layout 类型、convert_layout 代价、type-driven lowering | 35 min | 21, 23 |
| 25 | `25_phase4_register_occupancy.md` | ⭐ 寄存器分配与 Occupancy：三资源约束、spill 检测、num_warps/BLOCK/num_stages 权衡 | 30 min | 14, 22 |
| 26 | `26_phase4_ir_debugging.md` | IR 级性能调试方法论：4 种常见问题的诊断流程 + 速查表 | 25 min | 22, 24, 25 |

### 扩展视野

| # | 文件 | 内容 | 预计时间 | 前置知识 |
|---|------|------|----------|----------|
| 27 | `27_cutile_preview.md` | cuTile Python 预览：Layout 代数、MMA Atom | 20 min | 学完 Phase 1-2 后 |

## 怎么用

- **随 kernel 一起看**：建议先跑对应的 kernel，再回来看笔记
- **编译器视角**：标记 `> 🔧` 的段落是和 LLVM/MLIR 的类比，编译器背景读者重点关注
- **零基础友好**：00-02 从头讲起，不需要任何 GPU 前置知识
- **03 是精度选择**：`03_numerical_precision.md` 帮你决定什么时候用 fp16/bf16/fp8
- **08 是最实用的**：开发过程中遇到 bug 优先看 `08_debugging_triton.md`
- **10 是 autotune 策略**：`10_autotuning_strategy.md` 教你设计高效的搜索空间
- **13 是生产清单**：`13_production_checklist.md` 帮你从"能跑"到"生产可用"
- **15-17 是生产实战**：写完基础 kernel 后必读——block pointer、TMA、persistent kernel
- **21 是最重要的**：`21_triton_compiler_pipeline.md` 是整个项目的精华

---

## 待写内容

> 📋 下面是计划中但尚未完成的笔记，欢迎贡献。

### 高优先级（实战价值高）

| # | 主题 | 内容预告 | 为什么重要 |
|---|------|---------|-----------|
| 28 | **端到端实战：Transformer Block** | 把 GEMM + Flash Attention + LayerNorm + SwiGLU 串联成一个完整的 LLM 前向 pass，展示 kernel fusion 的端到端效果 | 学了那么多 kernel，是时候串起来了 |
| 29 | **性能剖析深入：ncu/nsys 完全指南** | 逐 section 解读 ncu 输出、Nsight Systems 时间线分析、Source/PTX/SASS 关联查看、Roofline 交互式分析 | 会看 profile 才能找到真正的瓶颈 |
| 30 | **LLM 推理优化专题** | KV cache 管理、Continuous Batching、量化部署（GPTQ/AWQ/GGUF）、Speculative Decoding 的 Triton 实现 | Triton 最热门的应用场景 |
| 31 | **Attention 变体实现** | GQA (Grouped Query Attention)、MQA (Multi-Query Attention)、Sliding Window Attention、PagedAttention | 现代 LLM 的核心注意力机制 |

### 中优先级（深入理解）

| # | 主题 | 内容预告 | 为什么重要 |
|---|------|---------|-----------|
| 32 | **GPU 架构演进史** | Kepler → Maxwell → Pascal → Volta → Ampere → Hopper → Blackwell，每一代的关键创新和编程模型变化 | 理解硬件设计决策，预测未来趋势 |
| 33 | **Triton 源码导读** | `lib/Dialect/`, `lib/Conversion/`, `python/triton/compiler/` 的代码结构和关键函数（`20_source_guide.py` 提供框架） | 想给 Triton 提 PR 或深入 debug 的必备 |
| 34 | **Sparse Computing 专题** | 2:4 结构化稀疏、Block Sparse、Sparse Flash Attention、Triton 的稀疏计算支持 | 未来方向——稀疏是通往更快计算的路径 |

### 低优先级（扩展视野）

| # | 主题 | 内容预告 | 为什么重要 |
|---|------|---------|-----------|
| 35 | **Triton vs TVM vs Halide vs MLIR** | 四个 DSL/编译器框架的对比：设计哲学、IR 设计、codegen 质量、适用场景 | 做编译器研究或选型时的参考 |
| 36 | **Debug 进阶：PTX/SASS 级别调优** | 手写 PTX、SASS 指令级优化、寄存器分配干预、cuobjdump 分析（`21_ptx_to_sass.py` 提供基础） | 追求最后 5-10% 性能的终极手段 |
| 37 | **cuTile Python 实战** | 用 NVIDIA cuTile Python 重写 GEMM，对比 Triton：性能、易用性、编译流程 | Phase 4 的核心内容 |
| 38 | **Triton Kernel 安全性** | 越界访问检测、UB (undefined behavior) 分析、race condition 排查、fuzzing | 生产环境部署前的最后防线 |

> 💡 **贡献方式**：挑一个主题，按现有笔记的格式（目标读者 + 新手友好 + 🔧 编译器视角 + 参考资料）写成 Markdown，放到 `notes/` 目录下。
