# Phase 2 — 核心计算内核

## 学习目标

掌握 GEMM 优化和 IO-aware 算法设计，理解 shared memory、software pipeline、autotuning。

## 文件列表

| # | 文件 | 状态 | 关键概念 | 难度 |
|---|------|------|----------|------|
| 1 | `01_matmul_naive.py` | ✅ 完成 | Basic tiling (M/N/K), TFLOPS 计算 | ⭐⭐ |
| 2 | `02_matmul_tiled.py` | ✅ 完成 | Shared memory, autotune, num_stages, num_warps | ⭐⭐⭐ |
| 3 | `03_matmul_autotuned.py` | ✅ 完成 | GROUP_M swizzling, L2 cache locality, 150 configs | ⭐⭐⭐ |
| 4 | `04_matmul_split_k.py` | ✅ 完成 | 3D grid, K-dim parallelism, atomic_add reduction | ⭐⭐⭐ |
| 5 | `05_matmul_fused_bias_act.py` | ✅ 完成 | Epilogue fusion, GELU/SiLU, 5× HBM savings | ⭐⭐⭐ |
| 6 | `06_matmul_transpose.py` | ✅ 完成 | 4 transpose variants (NN/NT/TN/TT), stride semantics | ⭐⭐ |
| 7 | `07_flash_attention_v1.py` | ✅ 完成 | Online softmax, IO-aware tiling | ⭐⭐⭐⭐ |
| 8 | `08_flash_attention_v2.py` | 📝 TODO | Causal masking, improved parallelism | ⭐⭐⭐⭐ |
| 9 | `09_depthwise_conv.py` | 📝 TODO | Convolution-to-GEMM mapping | ⭐⭐⭐ |
| 10 | `10_flash_attention_backward.py` | ✅ 完成 | dQ/dK/dV, LSE recompute, softmax backward | ⭐⭐⭐⭐⭐ |
| 11 | `11_grouped_query_attention.py` | ✅ 完成 | GQA/MQA, KV head sharing, KV cache reduction | ⭐⭐⭐ |
| 12 | `12_sliding_window_attention.py` | ✅ 完成 | Local attention, window mask, O(N·W) complexity | ⭐⭐⭐ |
| 13 | `13_attention_bias.py` | ✅ 完成 | ALiBi, vector/matrix bias, bias+softmax fusion | ⭐⭐⭐ |

## 运行方式

```bash
# 单个文件
python phase2_compute/01_matmul_naive.py

# 使用 Makefile
make run-matmul-naive
make run-matmul-tiled
make run-flash-v1
make run-flash-v2
make run-flash-bwd
make run-gqa
make run-sliding-window
make run-attention-bias

# 运行所有 Phase 2 kernels
make run-phase2
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

### `07_flash_attention_backward.py` — 必读

Flash Attention 反向传播：
- 从 saved LSE 重新计算 softmax 权重
- dQ/dK/dV 的 block-by-block 推导 (论文 Appendix B)
- torch.autograd.Function 集成

### `08_grouped_query_attention.py` — 推荐

GQA/MQA 的 Triton 实现：
- KV head sharing 的索引映射
- 降低 KV cache 大小的核心优化
- Llama-2/3、Mistral 的标配

### `09_sliding_window_attention.py` — 推荐

Mistral-7B 的滑动窗口注意力：
- Window mask 与 causal mask 的组合
- O(N·W) 计算复杂度
- NaN guard for fully-masked KV blocks

### `11_matmul_split_k.py` — 推荐

Split-K 并行化 GEMM：
- 3D grid 拆分 K 维
- atomic_add 归约部分和
- 适用于 K >> M,N 的场景

### `12_matmul_fused_bias_act.py` — 推荐

Transformer FFN epilogue fusion：
- MatMul + Bias + Activation 合一
- 省去 5× HBM 中间数据往返
- 支持 ReLU / GELU / SiLU

### `13_matmul_transpose.py` — 推荐

4种转置组合 (NN/NT/TN/TT)：
- Stride 操控实现零拷贝转置
- Coalesced access 分析
- NT 对应 Q @ K^T (attention 核心模式)

## 渐进学习路径

```
矩阵乘 (初学者路线):
  01_naive → 02_tiled+autotune → 03_swizzling+L2
                                   ↓
                              11_split-K (K 维并行)
                                   ↓
                              12_fused (epilogue 融合)
                                   ↓
                              13_transpose (转置变体)

注意力 (进阶路线):
  04_flash_v1 → 05_flash_v2 → 07_backward → 08_GQA → 09_sliding_window → 10_bias
```

## Benchmark

```bash
# 注意力 benchmark（包含所有新 kernel）
python benchmarks/bench_attention.py

# 快速模式
python benchmarks/bench_attention.py --quick
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
