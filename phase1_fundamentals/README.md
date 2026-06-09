# Phase 1 — Triton 基础

## 学习目标

掌握 Triton 的核心编程抽象，写出正确的 elementwise + reduction kernel。

## 文件列表

| # | 文件 | Kernel | 关键概念 | 难度 |
|---|------|--------|----------|------|
| 1 | `01_vector_add.py` | Vector Add | `@triton.jit`, `tl.program_id`, `tl.load/store`, autotune | ⭐ |
| 2 | `02_fused_softmax.py` | Fused Softmax | Reduction (`tl.max`, `tl.sum`), masking, fusion | ⭐⭐ |
| 3 | `03_fused_relu_bias.py` | Fused ReLU+Bias | Operator fusion, bandwidth analysis | ⭐ |
| 4 | `04_layer_norm.py` | Layer Norm | Welford 算法, shared memory reduction | ⭐⭐⭐ |

## 运行方式

```bash
# 单个文件
python phase1_fundamentals/01_vector_add.py
python phase1_fundamentals/02_fused_softmax.py
python phase1_fundamentals/03_fused_relu_bias.py
python phase1_fundamentals/04_layer_norm.py

# 使用 Makefile
make run-vector-add
make run-softmax
make run-relu-bias
make run-layernorm

# 全部运行
make run-phase1
```

每个文件可以直接 `python <file>` 运行，自带 `__main__` 测试代码，会输出：
- ✅/❌ 正确性验证（vs PyTorch reference）
- 性能对比（Triton vs PyTorch 时间/带宽）

## 测试

```bash
pytest tests/test_phase1.py -v
```

## 学习建议

1. 先跑 `01_vector_add.py`，对着代码理解每一行的 GPU 语义
2. 然后看 `notes/00_gpu_execution_model.md` 和 `notes/01_triton_programming_model.md`
3. 再跑 `02_fused_softmax.py`，理解 reduction
4. `04_layer_norm.py` 当前是简化版（3-pass），后续可优化为 Welford 1-pass
