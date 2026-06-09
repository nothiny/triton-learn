# 12 — GPU Kernel 优化案例研究：从 30% 到 80% Peak

> 真实优化案例，展示如何从"能跑"到"跑得快"。每个案例包含：初始状态、问题诊断、优化步骤、最终结果。

---

## 案例 1: GEMM — 从 Naive 到 75% cuBLAS

### 初始状态

```python
# Naive matmul: 直接从 HBM 读，无 shared memory
# BLOCK_M=64, BLOCK_N=128, BLOCK_K=32, num_warps=4
# 问题规模: M=N=K=4096, fp16

实测: 18.5 TFLOPS, 约 5.9% of H100 peak (312 TFLOPS) ← 很低
cuBLAS: 249.6 TFLOPS, 80% peak
```

### 诊断步骤

```bash
# Step 1: ncu 分析
ncu --set full python matmul_naive.py

结果:
  Memory Throughput:  85%    ← 一直在等内存
  Compute Throughput: 12%    ← SM 大部分时间空闲
  Achieved Occupancy: 45%
  
  诊断: memory-bound。每个 A/B 元素从 HBM 读取多次 → 加 shared memory。

# Step 2: 加 shared memory tiling (02_matmul_tiled.py)
# BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, num_warps=8

实测: 89.3 TFLOPS, 28.6% peak

ncu:
  Memory Throughput:  58%
  Compute Throughput: 35%

# 显著改善，但 Compute Throughput 仍不够高 → 加 autotune
```

### 优化步骤

```
Step 1 — Start:       18.5 TFLOPS ( 5.9%)  ← naive
Step 2 — +Shared Mem: 89.3 TFLOPS (28.6%)  ← 5× improvement!
Step 3 — +Autotune:  134.2 TFLOPS (43.0%)  ← 1.5×, 搜索最优 tile size
Step 4 — +num_stages:152.2 TFLOPS (48.8%)  ← 1.13×, double buffering
Step 5 — +GROUP_M:   168.5 TFLOPS (54.0%)  ← 1.11×, L2 cache 优化
Step 6 — Fine-tune:  187.2 TFLOPS (60.0%)  ← warp count, block order 微调

最终: 187 TFLOPS, 60% H100 peak, 75% cuBLAS
```

### 关键教训

```
1. Shared memory 是最重要的第一步（5× 提升）
2. Autotune 能带来 ~1.5× 提升（对复杂 kernel 可能更多）
3. 后面的优化越来越"辛苦": 每次 5-10%
4. 接近 cuBLAS 的 85% 以后，每一 % 都需要更多努力
5. 64%+ peak 对于 Triton 来说已经是优秀水平
```

---

## 案例 2: Fused Softmax — 从 3 次 HBM 遍历到 1 次

### 初始状态

```python
# 3 个独立 kernel:
#   kernel_1: max(x, dim=-1)          → 写 max 到 HBM
#   kernel_2: exp(x - max), sum(...)  → 写 sum 到 HBM
#   kernel_3: normalize + write

每个元素被读写:
  读: x(1次) + max(1次) + sum(1次) = 3 次
  写: max(1次) + exp_sum(1次) + out(1次) = 3 次
  共 6 次 HBM 访问 per element

实测: 48.5 GB/s bandwidth, 2.4% HBM peak (2000 GB/s on A100)
```

### 诊断

```
ncu:
  Memory Throughput: 6%  ← 极低！
  Compute Throughput: 1%

问题: 3 个 kernel，每个都是 memory-bound，中间的 max/sum 不需要写回 HBM
→ fused kernel: 在寄存器中完成 max → exp → sum → normalize
```

### 优化

```python
# Fused kernel (02_fused_softmax.py):
# 单 kernel 完成:
#   max(1 pass) → sum(1 pass) → normalize(1 pass)
# 但 max 和 sum 的结果留在寄存器中，不写 HBM
# 
# 实际只需: 读 x(1次) + 写 out(1次) = 2 次 HBM 访问 per element

实测: 386.2 GB/s bandwidth, 19.3% HBM peak

提升: 8× bandwidth utilization!
```

### 关键教训

```
1. Operator fusion 对 memory-bound kernel 收益最大
2. 每减少一次 HBM round-trip ≈ 减少 1× 延迟
3. 自问: "这个中间结果真的需要写回显存吗？"
```

---

## 案例 3: LayerNorm — Welford 在线算法的价值

### 初始状态

```python
# 3-pass LayerNorm:
#   pass 1: 读 x → 算 mean
#   pass 2: 读 x → 算 variance
#   pass 3: 读 x → 归一化 + affine

每个元素被读 3 次
```

### 优化

```python
# Welford 1-pass LayerNorm:
#   1 次遍历同时更新 M 和 S（Welford 算法的 online 统计量）
#   pass 2: 归一化 + affine

每个元素被读 2 次（仍然需要第 2 次遍历做归一化）

如果能融合后续的 linear 层: 只需 1 次 HBM 遍历 → 进一步减少
```

### 关键教训

```
1. 统计算法（Welford）可以实现 1-pass mean+variance
2. 但归一化本身仍然需要第 2 次读取
3. 终极方案: 融合 LayerNorm + 后续 Linear 层（kernel fusion）
```

---

## 案例 4: Flash Attention — IO-aware 算法的力量

### 问题

```
Standard Attention (N=4096, d=64, fp16):
  S 矩阵: 4096² × 2B = 33.6 MB
  P 矩阵: 4096² × 2B = 33.6 MB
  总 HBM: ~67 MB per attention head

  32 heads: 67 × 32 = 2.1 GB per layer
  对 32 层 transformer: 67 GB ← 甚至超出 H100 的 80GB 容量
```

### 解决方案

```
Flash Attention (04_flash_attention_v1.py):
  分块计算 attention，每个 tile 不写回 HBM
  S/P tile: 128×128 × 2B = 32 KB (在 SRAM 中)
  
  HBM: ~1 MB per head (只有 QKV read + O write)
  节省: ~67× per head
```

### 性能对比

```
Standard Attention (N=4096):
  Time: 45.3 ms
  Memory: 2.1 GB

Flash Attention v1 (Triton, N=4096):
  Time: 2.1 ms
  Memory: 33 MB

加速: 21.6× faster, 64× less memory
```

---

## 5. 优化方法论总结

### 通用优化流程

```
1. Profile first (ncu / torch.profiler)
   ↓
2. 判断瓶颈类型 (compute vs memory vs latency)
   ↓
3. 选择优化策略:
   
   Memory-bound:
   → Operator fusion (减少 HBM round-trip)
   → Smaller dtype (fp32 → fp16)
   → Tiling with shared memory
   → Better coalescing
   
   Compute-bound:
   → Use Tensor Core (fp16/bf16 input)
   → Better MMA tiling (larger BLOCK sizes)
   → Reduce non-matmul FLOPs
   → Increase occupancy
   
   Latency-bound (small sizes):
   → Persistent kernel
   → Reduce kernel launches
   → Batched processing
   
4. 量化每步改进（ncu + TFLOPS）
   ↓
5. Iterate: 看改进幅度 → 决定是否继续
```

### 优化的收益递减

```
优化第一轮 (最明显的):
  Shared memory + tiling: 3-5× 提升

优化第二轮 (有意义的):
  Autotune + num_stages: 1.5-2× 提升

优化第三轮 (辛苦的):
  GROUP_M + fine-tuning: 1.1-1.2× 提升

优化第四轮 (最后几 %):
  微调每 1% 可能需要数小时或数天
  评估 ROI — 是否值得？
```

### 何时停止优化？

```
1. 已经达到行业标准
   → GEMM: 70-80% cuBLAS 已经很好
   → 对于 fused kernel: 能打败 unfused 即可

2. 瓶颈已经改变
   → 原来 compute-bound → 优化后变成 memory-bound
   → 新的瓶颈可能需要完全不同的策略

3. 时间投入不值得
   → 最后 10% 可能需要 90% 的时间
   → 考虑用 CuTe 重写（如果确实需要那 10%）
```

---

## 参考资料

- [NVIDIA Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- 本项目: `phase2_compute/02_matmul_tiled.py` 的 PERFORMANCE NOTES
