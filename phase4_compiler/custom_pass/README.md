# Custom MLIR Pass

## Overview

Triton 的编译器管线基于 MLIR (Multi-Level Intermediate Representation)。
虽然 Triton 提供了完整的编译管线，理解 pass 机制有助于深入理解编译器工作原理。

## 概念

- **MLIR Pass**: 对 IR 进行转换的基本单元
- **Dialect**: 特定领域的操作和类型的集合（如 `tt`, `ttg`）
- **Conversion Pass**: 将一种 dialect 降低到另一种

## Triton Pass Pipeline

```
Python AST
  │
  ▼
TTIR (tt dialect)
  │ TritonInliner, TritonCombineOps
  ▼
TTIR (optimized)
  │ ConvertTritonToTritonGPU  ← 最关键的 pass
  ▼
TTGIR (tt + ttg dialect, with layout)
  │ TritonGPUPipeline, Prefetch, AccelerateMatmul, ...
  ▼
TTGIR (optimized)
  │ ConvertTritonGPUToLLVM
  ▼
LLVM IR
  │ LLVM NVPTX backend
  ▼
PTX
```

## 学习建议

1. 先读懂 `notes/03_triton_compiler_pipeline.md` 中每个 pass 的作用
2. 用 `01_dump_ir.py` 实际观察每个 pass 前后的 IR 变化
3. 阅读 Triton 源码中的 pass 定义: `triton/lib/Conversion/`, `triton/lib/Dialect/`

## References

- [MLIR Documentation](https://mlir.llvm.org/docs/)
- [Triton MLIR Dialect](https://github.com/triton-lang/triton/tree/main/lib/Dialect)
- [Writing a New MLIR Pass](https://mlir.llvm.org/docs/Tutorials/QuickStartRewrites/)
