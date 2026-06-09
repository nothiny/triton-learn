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

## 案例 5: RMSNorm — 去掉均值计算，但还有优化空间

### 初始状态

```python
# Naive RMSNorm: 2-pass
#   pass 1: 读 x → 算 rms = sqrt(mean(x²))
#   pass 2: 读 x → 归一化 + affine

# RMSNorm 比 LayerNorm 简单（不需要算 mean）

问题规模: N_ROWS=4096, N_COLS=4096
实测: 71.2 GB/s bandwidth, 3.6% H100 HBM peak
```

### 诊断

```
ncu 分析:
  Memory Throughput: 28%
  Compute Throughput: 2%
  
问题: 仍然是 memory-bound。2 次 HBM 遍历。
每个元素: 读 2 次 + 写 1 次 = 3 次 HBM 访问
```

### 优化步骤

```
Step 1 — 基础实现 (2-pass):
  71.2 GB/s, 3.6% peak

Step 2 — 融合 rms 计算和归一化到 1-pass:
  将 rms 的中间结果保存在寄存器中，不再写回 HBM
  但 normalization 仍需要第 2 次遍历
  
  结果: 105.8 GB/s, 5.3% peak (+47%)

Step 3 — 与后续操作融合（如果有 Linear 层）:
  将 RMSNorm + Linear 融合为一个 kernel
  → 完全消除 HBM round-trip
  → 这是 Liger Kernel 的关键优化
  
  Liger RMSNorm: ~380 GB/s, 19% peak
  (vs 我们的 2-pass: 71 GB/s)
```

### 关键教训

```
1. RMSNorm 比 LayerNorm 快约 1.3×（少算 mean）
2. 真正的优化在于 fusion: RMSNorm + Linear 融合 → 1 次 HBM 遍历
3. 单 kernel RMSNorm 的最大瓶颈是 HBM 带宽 — 无法突破
```

---

## 案例 6: SwiGLU — 从分离到融合

### 概念

```
SwiGLU(x) = SiLU(gate) ⊙ up

其中:
  gate = x @ W_gate.T    (Linear)
  up   = x @ W_up.T      (Linear)
  SiLU(z) = z * sigmoid(z)  (elementwise)

标准实现: 3 个 kernel
  1. Linear: gate = x @ W_gate.T
  2. Linear: up = x @ W_up.T
  3. SiLU + multiply: result = (gate * sigmoid(gate)) * up

问题: gate 和 up 写回 HBM → 再从 HBM 读回做 elementwise
```

### 优化

```python
# Fused SwiGLU: 单 kernel 完成 gate/up 的 GEMM + SiLU + multiply

@triton.jit
def fused_swiglu_kernel(
    x_ptr, w_gate_ptr, w_up_ptr, out_ptr,
    M, N, K,
    # ... strides, BLOCK sizes
):
    """
    result = (gate * sigmoid(gate)) * up
    其中 gate 和 up 在 kernel 内计算，不写回 HBM
    """
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # 两个累加器: gate 和 up
    acc_gate = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    acc_up = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        # Load x tile (复用于 gate 和 up)
        x = tl.load(x_ptr + offs_m[:, None] * K + 
                     (k + offs_k)[None, :], ...)
        
        # Load W_gate tile → compute gate
        w_g = tl.load(w_gate_ptr + offs_n[None, :] * K +
                       (k + offs_k)[:, None], ...)
        acc_gate += tl.dot(x, w_g)
        
        # Load W_up tile → compute up
        w_u = tl.load(w_up_ptr + offs_n[None, :] * K +
                       (k + offs_k)[:, None], ...)
        acc_up += tl.dot(x, w_u)
    
    # SiLU: z * sigmoid(z)
    # Triton 没有内置 sigmoid → 用 exp 实现
    sigmoid_gate = 1.0 / (1.0 + tl.exp(-acc_gate))
    silu = acc_gate * sigmoid_gate
    
    # Elementwise multiply
    result = silu * acc_up
    
    tl.store(out_ptr + ..., result, ...)


# 对比:

# 标准实现:
#   gate = linear(x, w_gate)         # GEMM → write HBM (M×N fp16)
#   up = linear(x, w_up)             # GEMM → write HBM (M×N fp16)
#   result = silu(gate) * up          # read gate, read up → write HBM (M×N fp16)
#   总 HBM 流量: ~5 × M×N × 2 bytes

# Fused:
#   result = fused_swiglu(x, w_gate, w_up)
#   总 HBM 流量: 1 × (M×K) + 2 × (K×N) + 1 × (M×N) ≈ 远小于 unfused
```

### 性能对比

```
标准实现 (3 kernels):
  Time: 1.89 ms
  HBM traffic: ~50 MB

Fused SwiGLU:
  Time: 0.95 ms
  HBM traffic: ~25 MB

加速: 2.0×（主要来自减少了 HBM round-trip）
```

### Liger Kernel 对比

```
Liger 的 SwiGLU:
  - 用 chunked 策略: 把大矩阵切分成多个 chunk
  - 每个 chunk 独立做 GEMM + SiLU + multiply
  - 进一步减少显存峰值
  
  Liger SwiGLU 比我们的 fused 版本快 ~1.2×
  （更好的 tiling 和 register management）
```

### 关键教训

```
1. 对于激活函数（SiLU, GELU, ReLU）:
   → 总是在 GEMM 后面紧接着做 elementwise op
   → 节省一次 HBM round-trip

2. 当两个 GEMM 共享同一个输入 x:
   → 可以将它们融合为一个 kernel
   → x 只读一次 → 减少数据搬运

3. 内存带宽是扩展瓶颈:
   → fused kernel 的收益来自减少 HBM 流量
   → 对 memory-bound 的操作特别有效
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
