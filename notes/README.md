# Notes — 学习笔记

## 阅读顺序

| # | 文件 | 内容 | 预计时间 | 前置知识 |
|---|------|------|----------|----------|
| 1 | `00_gpu_execution_model.md` | GPU 执行模型：SM/warp/内存层级/Tensor Core | 30 min | 编译器后端经验 |
| 2 | `01_triton_programming_model.md` | Triton 编程模型：program 抽象、核心 API | 30 min | 00 |
| 3 | `02_memory_hierarchy.md` | 内存层级：coalescing/bank conflict/software pipeline | 30 min | 00 |
| 4 | `03_triton_compiler_pipeline.md` | ⭐ 编译器管线：TTIR→TTGIR→LLVM→PTX | 1 hr | 01 + MLIR 基础 |
| 5 | `04_cute_preview.md` | CuTe 预览：Layout 代数、MMA Atom | 20 min | 学完 Phase 1-2 后 |

## 怎么用

- **随 kernel 一起看**：建议先跑对应的 kernel，再回来看笔记
- **编译器视角**：标记 `> 🔧` 的段落是和 LLVM/MLIR 的类比，编译器背景读者重点关注
- **03 是最重要的**：如果你是编译器背景，`03_triton_compiler_pipeline.md` 是整个项目的精华
