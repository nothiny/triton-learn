# Notes — 学习笔记

## 阅读顺序

### 基础篇（先读这些）

| # | 文件 | 内容 | 预计时间 | 前置知识 |
|---|------|------|----------|----------|
| 1 | `00_gpu_execution_model.md` | GPU 执行模型：SM/warp/内存层级/Tensor Core | 45 min | 无，零基础可读 |
| 2 | `01_triton_programming_model.md` | Triton 编程模型：program 抽象、核心 API | 30 min | 00 |
| 3 | `02_memory_hierarchy.md` | 内存层级：coalescing/bank conflict/software pipeline | 30 min | 00 |
| 4 | `06_debugging_triton.md` | 🐛 调试 Triton Kernel：工具、常见 Bug、工作流 | 30 min | 01 |

### 核心篇（理解最重要）

| # | 文件 | 内容 | 预计时间 | 前置知识 |
|---|------|------|----------|----------|
| 5 | `03_triton_compiler_pipeline.md` | ⭐ 编译器管线：TTIR→TTGIR→LLVM→PTX | 1 hr | 01 + MLIR 基础 |
| 6 | `07_flash_attention_math.md` | ⭐ Flash Attention 完整数学推导 | 45 min | 01 |

### 进阶篇（深入优化）

| # | 文件 | 内容 | 预计时间 | 前置知识 |
|---|------|------|----------|----------|
| 7 | `08_gemm_advanced.md` | GEMM 优化进阶：warp tiling/pipeline/swizzling | 45 min | 00-03 |
| 8 | `09_tensor_core_deep.md` | Tensor Core 深入：MMA 指令、FP8、sparsity | 30 min | 00, 03 |
| 9 | `12_optimization_cases.md` | 真实优化案例：从 30% 到 80% peak | 30 min | 02, 08 |

### 扩展篇（架构与迁移）

| # | 文件 | 内容 | 预计时间 | 前置知识 |
|---|------|------|----------|----------|
| 10 | `10_hopper_architecture.md` | Hopper (H100) 架构：wgmma/TMA/FP8/cluster | 30 min | 00, 09 |
| 11 | `11_cuda_to_triton.md` | CUDA → Triton 迁移指南：概念映射、代码翻译 | 20 min | 01 + CUDA 经验 |
| 12 | `14_kernel_patterns.md` | 常见 Kernel 模式：reduce/scan/gather/conv | 30 min | 01, 02 |

### 参考篇（按需查阅）

| # | 文件 | 内容 | 预计时间 | 前置知识 |
|---|------|------|----------|----------|
| 13 | `13_triton_internals.md` | Triton 内部：JIT/Cache/driver API | 20 min | 03 |
| 14 | `04_cute_preview.md` | CuTe 预览：Layout 代数、MMA Atom | 20 min | 学完 Phase 1-2 后 |
| 15 | `05_benchmarking_methodology.md` | Benchmark 方法论：正确测量 GPU 性能 | 30 min | 01 |
| 16 | `15_multi_gpu.md` | Multi-GPU：NVLink/all-reduce/tensor parallelism | 20 min | 01 |

## 怎么用

- **随 kernel 一起看**：建议先跑对应的 kernel，再回来看笔记
- **编译器视角**：标记 `> 🔧` 的段落是和 LLVM/MLIR 的类比，编译器背景读者重点关注
- **零基础友好**：00-02 从头讲起，不需要任何 GPU 前置知识
- **03 是最重要的**：`03_triton_compiler_pipeline.md` 是整个项目的精华
- **06 是最实用的**：开发过程中遇到 bug 优先看 `06_debugging_triton.md`
