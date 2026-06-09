# 14 — 常见 Kernel 模式：Reduce、Scan、Gather、Convolution

> 除了 elementwise 和 GEMM，GPU 上还有很多常见的计算模式。这篇整理了在 Triton 中实现这些模式的惯用写法。

---

## 1. Reduce（归约）— 多对一的聚合

### 1.1 概念

```
Reduce: 把多个元素合并为一个（或少数几个）

常见 reduce:
  sum(x):   x₁ + x₂ + ... + xₙ
  max(x):    max(x₁, x₂, ..., xₙ)
  argmax(x): 最大值的索引
  mean(x):  sum(x) / n

Reduce 的本质: "沿某个维度聚合，产出更小的结果"
```

### 1.2 Triton 实现：Row-wise Softmax

```python
# 每行独立做 softmax — 每行是一个 reduce group
@triton.jit
def reduce_kernel(x_ptr, out_ptr, N_COLS, BLOCK_SIZE: tl.constexpr):
    """
    逐行 softmax: 沿列维（dim=1）做 reduce
    """
    row_idx = tl.program_id(0)  # 每个 program 处理一行
    
    # Step 1: find max（沿列维 reduce，用 max）
    row_max = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
    for block_start in range(0, N_COLS, BLOCK_SIZE):
        offsets = row_idx * N_COLS + block_start + tl.arange(0, BLOCK_SIZE)
        mask = (block_start + tl.arange(0, BLOCK_SIZE)) < N_COLS
        x = tl.load(x_ptr + offsets, mask=mask, other=float('-inf'))
        row_max = tl.maximum(row_max, x)  # per-block max
    global_max = tl.max(row_max, axis=0)  # final reduce across blocks
    
    # Step 2: compute sum（沿列维 reduce，用 sum）
    row_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for block_start in range(0, N_COLS, BLOCK_SIZE):
        offsets = row_idx * N_COLS + block_start + tl.arange(0, BLOCK_SIZE)
        mask = (block_start + tl.arange(0, BLOCK_SIZE)) < N_COLS
        x = tl.load(x_ptr + offsets, mask=mask, other=float('-inf'))
        row_sum += tl.exp(x - global_max)
    global_sum = tl.sum(row_sum, axis=0)
    
    # Step 3: normalize + write
    for block_start in range(0, N_COLS, BLOCK_SIZE):
        offsets = row_idx * N_COLS + block_start + tl.arange(0, BLOCK_SIZE)
        mask = (block_start + tl.arange(0, BLOCK_SIZE)) < N_COLS
        x = tl.load(x_ptr + offsets, mask=mask, other=float('-inf'))
        result = tl.exp(x - global_max) / global_sum
        tl.store(out_ptr + offsets, result, mask=mask)

# [COMPILER] tl.max(x, axis=0) 编译为:
# 1. warp shuffle reduction (fast, in-register)
# 2. shared memory reduction (cross-warp, slower)
```

### 1.3 Reduce 的性能要点

```
1. Reduce 通常是 memory-bound（很少 FLOPs per element）
   → 优化方向: 减少 HBM 遍历次数

2. tl.sum(axis=0) 跨 block 需要 shared memory
   → BLOCK_SIZE 太大意味着更多 reduce 开销

3. 对于大 reduction，可以考虑:
   - 多级 reduce（block-level → warp-level）
   - 使用 shared memory 缓存中间结果
   - 与后续 op 融合（如 fused softmax）
```

---

## 2. Scan（扫描）— 前缀和与累积

### 2.1 概念

```
Scan (prefix sum / cumulative sum):

Input:  [a₁, a₂, a₃, a₄, a₅]
Output: [a₁, a₁+a₂, a₁+a₂+a₃, a₁+a₂+a₃+a₄, a₁+a₂+a₃+a₄+a₅]

应用:
  - LayerNorm 的 mean/variance 计算
  - 排序、基数排序
  - Attention 的 causal masking
```

### 2.2 Triton 实现：Parallel Prefix Sum

```python
@triton.jit
def inclusive_scan_kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    Block-level inclusive scan (prefix sum).
    使用 Blelloch scan 算法: O(log n) steps, O(n) work
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Blelloch scan: 上扫 (up-sweep) + 下扫 (down-sweep)
    # Up-sweep: 构建部分和树
    for offset in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        # 从相邻元素累加
        prev = tl.where(tl.arange(0, BLOCK_SIZE) >= offset,
                        x, tl.zeros_like(x))
        # 需要 shift: shift right by offset
        # (简化版 — 实际需要 tl.shift_right 或等效操作)
        x = x + prev  # incomplete — 示意
    
    # [COMPILER] Triton 对 scan 的支持不如 reduce 原生
    # 因为 scan 有依赖链，不能完全并行
    # 通常用 warp shuffle 实现
    
    tl.store(out_ptr + offsets, x, mask=mask)
```

### 2.3 Scan 的 Triton 限制

```
Scan 在 Triton 中比较难实现的原因:
1. 有数据依赖（每个元素依赖前一个元素）
2. Triton 的 block-level API 不直接支持这种依赖
3. 通常需要用 warp shuffle 或 shared memory 实现

替代方案:
- 如果 scan 长度不大（<1024），用 PyTorch torch.cumsum 即可
- 如果需要大 scan，考虑分段 scan（decompose into independent blocks）
```

---

## 3. Gather / Scatter — 非连续的内存访问

### 3.1 概念

```
Gather:  根据索引数组从源数组中收集数据
  output[i] = input[index[i]]

Scatter: 根据索引数组向目标数组分发数据
  output[index[i]] = input[i]

应用:
  - Embedding lookup (gather)
  - Attention 的 KV cache 更新 (scatter)
  - Sparse matrix operations
```

### 3.2 Triton 实现：Embedding Lookup (Gather)

```python
@triton.jit
def gather_kernel(input_ptr, index_ptr, output_ptr,
                  N, EMBED_DIM,
                  BLOCK_SIZE: tl.constexpr):
    """
    Embedding lookup: output[i, :] = input[index[i], :]
    
    Gather 在 GPU 上容易实现但性能差:
    - index 可以指向 input 的任意位置
    - 无法做 coalescing
    - L2 cache 命中率取决于 index 的分布（随机 → 低，顺序 → 高）
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # 加载索引
    indices = tl.load(index_ptr + offsets, mask=mask, other=0)
    
    # Gather: 根据索引从 input 加载
    # 每个线程访问不同的 input 行 → 非合并访问 → 慢
    for d in range(0, EMBED_DIM, BLOCK_SIZE):
        d_offsets = d + tl.arange(0, BLOCK_SIZE)
        d_mask = d_offsets < EMBED_DIM
        
        # 计算每个线程要访问的地址
        ptrs = input_ptr + indices[:, None] * EMBED_DIM + d_offsets[None, :]
        vals = tl.load(ptrs, mask=mask[:, None] & d_mask[None, :])
        
        out_ptrs = output_ptr + offsets[:, None] * EMBED_DIM + d_offsets[None, :]
        tl.store(out_ptrs, vals, mask=mask[:, None] & d_mask[None, :])

# Performance note:
# - 随机 gather: ~10-20% HBM bandwidth（最坏情况）
# - 顺序 gather: ~80% HBM bandwidth（接近 coalesced）
```

### 3.3 Triton 实现：Scatter

```python
@triton.jit
def scatter_kernel(input_ptr, index_ptr, output_ptr,
                   N, BLOCK_SIZE: tl.constexpr):
    """
    Scatter: output[index[i]] = input[i]
    
    问题: 多个线程可能写同一个 output 位置 → 需要 atomic
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    indices = tl.load(index_ptr + offsets, mask=mask, other=0)
    
    # Scatter with atomic add（如果可能有冲突）
    out_ptrs = output_ptr + indices
    tl.atomic_add(out_ptrs, vals, mask=mask)
```

---

## 4. Convolution — 把卷积映射到 GEMM

### 4.1 三种实现策略

```
1. Direct convolution:
   直接写 conv 的 7 层嵌套循环
   优点: 简单，内存友好
   缺点: 难优化，cache 不友好

2. Im2col + GEMM:
   im2col: 把 (C, H, W) 展开为 (C*K*K, H_out*W_out) 矩阵
   GEMM: C_out × (C*K*K) @ (C*K*K) × (H_out*W_out)
   优点: 可以用高度优化的 GEMM
   缺点: im2col 有内存膨胀（K*K×）

3. Winograd / FFT:
   用数学变换减少乘法次数
   优点: 对特定 kernel size 极致优化
   缺点: 复杂，有精度问题
```

### 4.2 Triton 中实现 Depthwise Conv

```python
# 见 phase2_compute/06_depthwise_conv.py
# 关键: 每个 program 处理一个输出 spatial tile + 一组 channels
# 沿 KH×KW 维做 reduction

# 性能: 通常 memory-bound（算术强度 = KH*KW / (KH*KW+4) ≈ 0.5-0.9 FLOP/byte）
```

---

## 5. Atomic Operations — 处理并发写

### 5.1 Triton 支持的 Atomic

```python
tl.atomic_add(ptr, val)     # 原子加法
tl.atomic_max(ptr, val)     # 原子最大值
tl.atomic_min(ptr, val)     # 原子最小值
tl.atomic_and(ptr, val)     # 原子按位与
tl.atomic_or(ptr, val)      # 原子按位或
tl.atomic_xor(ptr, val)     # 原子按位异或
tl.atomic_cas(ptr, cmp, val) # 原子比较并交换
tl.atomic_xchg(ptr, val)    # 原子交换
```

### 5.2 Atomic 的性能代价

```
当多个线程对同一地址做 atomic 操作时:
  1 个线程: 正常速度
  2-4 个线程: 轻微变慢
  32+ 个线程: 严重串行化 → 可能比非 atomic 版本慢 10-100×

避免大量 atomic 竞争的策略:
  1. 先做 local reduce，最后再 atomic 写一次
  2. 用 tiling 减少 atomic 的粒度
  3. 如果可以，用 shared memory buffer 先缓冲再写
```

---

## 6. 常见模式的决策树

```
你想实现什么？

├── Elementwise: x[i] = f(y[i], z[i])
│   → 最简单的 kernel，memory-bound
│   → 考虑: 是否可以和相邻 op 融合？

├── Reduce: result = reduce(x, dim=d)
│   → memory-bound, 用 tl.sum/tl.max(axis=d)
│   → 考虑: 是否可以融入 producer/consumer kernel？

├── GEMM: C = A @ B
│   → 通常 compute-bound (大尺寸)
│   → 用 shared memory + autotune + num_stages
│   → 考虑: split-K 如果 K 很大

├── Scan: output[i] = f(output[i-1], input[i])
│   → 有数据依赖，难以并行
│   → 对小 scan (<1024): 用 PyTorch
│   → 对大 scan: 分段 scan 或考虑用 warp shuffle

├── Gather/Scatter:
│   → 通常 memory-bound, 合并访问难
│   → 随机 gather: 性能受限，接受就好

└── Attention:
    → memory-bound (长序列) 或 compute-bound (短序列)
    → 用 Flash Attention tiling
    → 可叠加 causal mask, GQA
```

---

## 参考资料

- [Triton Official Tutorials](https://triton-lang.org/main/getting-started/tutorials/)
- [GPU Gems 2 — Parallel Prefix Sum (scan)](https://developer.nvidia.com/gpugems/gpugems2/part-iv-general-purpose-computation-gpus-primer/chapter-39-parallel-prefix-sum-scan-cuda)
- [CUTLASS — Implicit GEMM for Convolution](https://github.com/NVIDIA/cutlass/blob/main/media/docs/implicit_gemm_convolution.md)
