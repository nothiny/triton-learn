# Phase 2 — 核心计算内核

## 学习目标

掌握 GEMM 优化和 IO-aware 算法设计，理解 shared memory、software pipeline、autotuning。

## 文件列表

| # | 文件 | 状态 | 关键概念 | 难度 |
|---|------|------|----------|------|
| 1 | `01_matmul_naive.py` | ✅ 完成 | Tiling (M/N/K), TFLOPS 计算 | ⭐⭐ |
| 2 | `02_matmul_tiled.py` | ✅ 完成 | Shared memory, autotune, num_stages, num_warps | ⭐⭐⭐ |
| 3 | `03_matmul_autotuned.py` | 📝 TODO | GROUP_M swizzling, 更大搜索空间 | ⭐⭐⭐ |
| 4 | `04_flash_attention_v1.py` | ✅ 完成 | Online softmax, IO-aware tiling | ⭐⭐⭐⭐ |
| 5 | `05_flash_attention_v2.py` | 📝 TODO | Causal masking, 改进并行化 | ⭐⭐⭐⭐ |
| 6 | `06_depthwise_conv.py` | 📝 TODO | 卷积→GEMM 映射 | ⭐⭐⭐ |

## 运行方式

```bash
# 单个文件
python phase2_compute/01_matmul_naive.py
python phase2_compute/02_matmul_tiled.py
python phase2_compute/04_flash_attention_v1.py

# 使用 Makefile
make run-matmul-naive
make run-matmul-tiled
make run-matmul-autotuned
make run-flash-v1
```

## 重点文件

### `02_matmul_tiled.py` — 必读

这是最完整的内核，展示了 Triton 的生产级 GEMM：
- `@triton.autotune` 多维度搜索
- Shared memory staging（编译器自动插入）
- `num_stages` software pipelining
- `num_warps` 控制 occupancy
- PERFORMANCE NOTES 中有 roofline 分析

### `04_flash_attention_v1.py` — 必读

Flash Attention 论文的 Triton 实现：
- Online softmax 算法（running max/sum rescaling）
- Block-by-block tiling（只需 O(sqrt(N)) SRAM）
- 注释对应论文 Algorithm 1 的每一行

## Benchmark

```bash
# 单独跑 GEMM benchmark
python benchmarks/bench_runner.py --category gemm

# 带 profiler
python benchmarks/bench_runner.py --category gemm --profile
```

## Profile

```bash
# ncu 深入分析
ncu --set full -o reports/matmul_tiled python phase2_compute/02_matmul_tiled.py

# 或使用 Makefile
make profile-matmul
```

## 学习建议

1. 先搞懂 `01_matmul_naive.py`（tiling 概念）
2. 再读 `notes/02_memory_hierarchy.md`（shared memory + pipeline）
3. 仔细研究 `02_matmul_tiled.py`（生产级 GEMM）
4. `04_flash_attention_v1.py` 是算法+硬件协同设计的最佳案例
