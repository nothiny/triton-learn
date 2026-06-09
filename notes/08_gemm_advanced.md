# 08 — GEMM 优化进阶：从 Tiled 到生产级

> **目标**: 理解从 naive matmul 到接近 cuBLAS 性能的每一步优化原理。
> **前置**: Phase 2 的 01/02/03_matmul，笔记 02（内存层级）

---

## 0. 优化路线图

```
Naive MatMul:                     HBM 原始读写，无 shared memory
    │                              性能: ~5-10% peak
    ▼
Tiled + Shared Memory:            使用 shared memory 缓存 A/B tile
    │                              性能: ~20-30% peak
    ▼
Autotuning:                       搜索最优 BLOCK_M/BLOCK_N/BLOCK_K/num_warps
    │                              性能: ~30-50% peak
    ▼
Warp Tiling:                      每个 warp 计算更大的 tile
    │                              性能: ~40-60% peak
    ▼
Double Buffering (num_stages=2):  加载和计算重叠
    │                              性能: ~50-70% peak
    ▼
GROUP_M Swizzling:                改进 L2 cache 局部性
    │                              性能: ~55-75% peak
    ▼
Register Banking + Prefetch:      寄存器级别的优化
    │                              性能: ~60-80% peak
    ▼
cuBLAS-level:                     手写 SASS、warp specialization (CuTe)
                                  性能: ~80-85% peak
```

---

## 1. 从 Naive 到 Tiled — Shared Memory 的价值

### 1.1 Naive MatMul 的问题

```python
# Naive: 直接从 HBM 读 A 和 B
for k in range(0, K, BLOCK_K):
    a = tl.load(a_ptr + offsets)  # 读一次
    b = tl.load(b_ptr + offsets)  # 读一次
    acc += tl.dot(a, b)

# 问题分析:
# A 和 B 的每个元素被读了多少次？
# A[BLOCK_M, BLOCK_K] 的每个元素 → 被复用 0 次（只在一个 dot 中使用）
# 实际上每个 A 元素要被复用 N/BLOCK_N 次才对！
# 
# naive 版本每次循环都从 HBM 重新读数据 → 大量重复 HBM 访问
```

### 1.2 Tiled 版本的改进

```python
# Tiled: 先把数据搬到 shared memory
for k in range(0, K, BLOCK_K):
    # Load A[BLOCK_M, BLOCK_K] to shared memory (implicit via tl.load)
    # Load B[BLOCK_K, BLOCK_N] to shared memory
    a = tl.load(a_ptr + offsets)  # Triton 自动 staging 到 shared memory
    b = tl.load(b_ptr + offsets)
    acc += tl.dot(a, b)

# A 的每个元素现在被复用 BLOCK_N/1 次（但对 N 维的所有计算都用到同一个 A tile）
# 实际上: shared memory 的存在让 A tile 的每个元素在计算
# C[BLOCK_M, BLOCK_N] tile 时被使用了 BLOCK_N 次（每个 C 列一次）
```

### 1.3 量化收益

```
以 (BLOCK_M=128, BLOCK_N=128, BLOCK_K=32) 为例:

Naive HBM 访问（一个 C tile）:
  for each k in K/BLOCK_K:  # K=4096 → 128 iterations
    读 A: 128×32 = 4096 elements × 2 bytes = 8 KB
    读 B: 32×128 = 4096 elements × 2 bytes = 8 KB
  总 = 128 × (8+8) = 2048 KB per C tile

Tiled HBM 访问:
  同样: 2048 KB per C tile（加载到 shared memory 是一样的）
  BUT: 从 shared memory 读取的带宽是 HBM 的 ~10×

  所以实际加速来自:
  1. 更低的延迟（shared memory 20-30 cycles vs HBM 500 cycles）
  2. 更高的带宽（shared memory 更高 BW）
  3. A tile 在 shared memory 中被 K 维循环反复使用
     - 对于整个 K 维循环: 数据只加载一次到 shared memory
     - 之后的每次 K 迭代: 新数据覆盖旧的 shared memory
```

---

## 2. Warp Tiling — 让每个 warp 做更多

### 2.1 概念

```
基础 tiling (BLOCK_M=128, BLOCK_N=128):
  每个 block 的 8 个 warp 共享这个 C tile
  每个 warp 负责 C tile 的一部分
  
  问题: 每个 warp 的计算量有限
    → warp 可能在等待 sync/load 时空闲
    → Tensor Core 利用率不高

Warp Tiling (BLOCK_M=128, BLOCK_N=128, warp_tile_M=64, warp_tile_N=64):
  每个 warp 负责 64×64 = 4096 个元素的子 tile
  而不是 128×128/8 = 2048 个元素分配方式
  
  好处:
  - 每个 warp 有更多独立计算 → 减少 warp 间同步
  - 更大的 warp tile → 更好的 Tensor Core 利用率
  - 每个 warp 可以在自己的 tile 上连续做 MMA
```

### 2.2 在 Triton 中实现 Warp Tiling

Triton 通过 `sizePerThread` 和 `threadsPerWarp` 在 layout 中隐式控制 warp tiling。你不需要显式写 warp-level 代码——通过调整 `BLOCK_M`, `BLOCK_N` 和 `num_warps` 间接控制：

```python
# 小 warp tile（细粒度，多同步）
triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8)
# 8 warps 共享 128×128 tile → 每个 warp 平均 ~2048 元素

# 大 warp tile（粗粒度，少同步，但可能 occupancy 低）
triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256}, num_warps=8)
# 8 warps 共享 256×256 tile → 每个 warp 平均 ~8192 元素
```

---

## 3. Software Pipelining — 让加载和计算同时进行

### 3.1 问题可视化

```
没有 pipeline (num_stages=1):
  Warp: |-- Load A[0] --|-- Compute[0] --|-- Load A[1] --|-- Compute[1] --|
        |               |                |               |                |
  SM:   [Memory Busy   ][Compute Busy   ][Memory Busy   ][Compute Busy   ]
        但 Memory 和 Compute 是独立的硬件单元 → 其中一个忙时另一个空闲！

有 pipeline (num_stages=2):
  Warp: |-- Load A[0] --|-- Load A[1] --|-- Load A[2] --|
        |               |-- Compute[0] -|-- Compute[1] -|-- Compute[2] -|
  
  SM:   [Mem Busy       ][Mem Busy      ][Mem Busy      ]
        [               ][Comp Busy     ][Comp Busy     ][Comp Busy     ]
        
  → Memory 和 Compute 几乎 100% 重叠 → 约 2× 吞吐提升
```

### 3.2 `num_stages` 选择指南

```python
# num_stages=1: 同步模式
#   加载全部数据 → 计算 → 加载下一批 → ...
#   代码简单，调试方便
#   Shared memory: 1× tile
#   适用: shared memory 紧张时

# num_stages=2: double buffering
#   两个 buffer: buffer[0] 在计算时，buffer[1] 在加载
#   最常用的模式
#   Shared memory: 2× tile
#   适用: 大多数情况

# num_stages=3-4: 更深的流水线
#   更大延迟隐藏能力
#   Shared memory: 3-4× tile
#   可能触发 shared memory 溢出 → occupancy 下降 → 得不偿失

# 经验法则: 从 num_stages=2 开始，如果你的 shared memory 够用且 kernel 是 memory bound，尝试 3。
```

### 3.3 在 Triton 中的实现

```python
@triton.jit
def matmul_kernel(..., BLOCK_M, BLOCK_N, BLOCK_K):
    # 你不需要写任何 pipeline 代码！
    # Triton 编译器自动:
    # 1. 识别 K 维循环是 pipeline 候选
    # 2. 展开循环
    # 3. 插入 cp.async 指令做异步 global→shared 拷贝
    # 4. 插入 commit_group + wait_group 做异步同步
    
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptr + ...)  # 编译器变成 cp.async
        b = tl.load(b_ptr + ...)  # 编译器变成 cp.async
        acc += tl.dot(a, b)       # 编译器插入 wait 确保数据到了
    # 只需设 num_stages=2，以上自动发生
```

---

## 4. GROUP_M Swizzling — L2 Cache 优化

### 4.1 问题

```
默认 grid 调度:
  假设 M=4096, N=4096, BLOCK_M=128, BLOCK_N=128
  num_pid_m = 32, num_pid_n = 32
  
  默认调度: pid = pid_m × num_pid_n + pid_n  (row-major)
  pid=0:  (pid_m=0,  pid_n=0)   → C[0:128, 0:128]
  pid=1:  (pid_m=0,  pid_n=1)   → C[0:128, 128:256]
  ...
  pid=32: (pid_m=1,  pid_n=0)   → C[128:256, 0:128]
  
  问题: 相邻 pid 处理同一 A 行但不同 B 列
    B 列不同 → B tile 在内存中连续 → L2 友好（无需担心）
    但 A tile: pid=0..31 都用 A[0:128, :] — 反复从 HBM 加载！
    
  好的做法: 让相邻 pid 处理同一 A tile
  → pid=0: C[0:128, 0:128]   用 A[0:128, :]
  → pid=1: C[0:128, 256:384]  也用 A[0:128, :]  ← A tile 复用！
```

### 4.2 GROUP_M 的解决方案

```python
# GROUP_M swizzling: 重新映射 program_id
# 将 M 维度分成 GROUP_M 个一组，组内按列优先

GROUP_M = 8  # 8 个 M block 为一组

组 0: pid_m ∈ {0, 1, ..., 7} 对 pid_n 的映射:
  pid=0:  (0, 0)   → 先用所有 N 列，再用 A[0:128, :]
  pid=1:  (1, 0)   → 但仍然用 A[0:128+:256, :]
  ...等等

# 关键 insight:
# 每 GROUP_M 个 M block 中，它们共用相似的列索引
# 当 pid_n 在组内重复时，A 的某些行被多个 pid 共享 → L2 命中率提升

# 在 Triton 中:
@triton.autotune(configs=[triton.Config({'BLOCK_M': m, ..., 'GROUP_M': g}, ...)
                            for g in [1, 4, 8]], key=['M', 'N', 'K'])
@triton.jit
def matmul_kernel(..., GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_in_group = pid % num_pid_in_group
    
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m
```

### 4.3 GROUP_M 的收益

```
GROUP_M=1 (默认): 
  L2 cache 命中率 ~30-40%

GROUP_M=4:
  L2 cache 命中率 ~50-60%
  适合: M=1024-4096 的中等规模

GROUP_M=8:
  L2 cache 命中率 ~60-70%
  适合: M=2048-8192
  
小规模 (M<256): GROUP_M 意义不大，L2 容量足够
超大规模 (M>16384): GROUP_M 帮助有限，每个 A tile 只被少量 pid 复用
```

---

## 5. Split-K — 把 K 维也并行化

### 5.1 问题

```
标准 tiling: grid 沿 M 和 N 维度并行
  grid = (num_pid_m, num_pid_n)
  
  问题: 如果 K 很大 (K > 16384), K 维循环很长
  所有 pid 都要遍历完整的 K 维 → 每个 pid 工作时间长
  → 如果 pid 不多 (M/N 很小), GPU 上很多 SM 空闲
```

### 5.2 Split-K 的解决方案

```python
# Split-K: 把 K 维也切成多个块，每个块独立计算部分结果，最后求和

grid = (num_pid_m, num_pid_n, num_pid_k)  # 3D grid!

# 每个 pid_k 只处理 K 维的一部分:
pid_k = tl.program_id(2)
k_per_split = tl.cdiv(K, num_pid_k)
k_start = pid_k * k_per_split
k_end = min(k_start + k_per_split, K)

# 每个 pid 计算部分结果:
acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
for k in range(k_start, k_end, BLOCK_K):
    ...
# 最后用 atomic_add 把部分结果累加到最终输出
tl.atomic_add(c_ptr + offsets, acc, mask=mask)
# 但 atomic_add 有性能开销 → 需要 tradeoff
```

**Split-K 的 tradeoff**: 更多并行 → 但需要 atomic_add → 有同步开销。通常只在 K 特别大或 M/N 特别小的时候有用。

---

## 6. Persistent Kernel — 减少 Launch Overhead

### 6.1 问题

```
普通 kernel: 每个 block 处理一个 tile
  block 数量 = num_pid_m × num_pid_n
  
  问题: 每个 block 是独立 launch 的
  → 如果 tile 很多但每个 tile 很小 → launch overhead 主导

Persistent kernel:
  固定数量的 block (如 num_SMs)，每个 block 持续 grab 下一个 tile:
  
  while True:
    tile_idx = atomic_add(&global_counter, 1)  # 取下一个 tile
    if tile_idx >= total_tiles: break
    compute_tile(tile_idx)
  
  好处: 
  - 只有一次 launch
  - 自动负载均衡（快的 block 多做几个 tile）
  
  缺点:
  - 需要 atomic 操作
  - 不再有 autotuner 的 grid-level 便利
```

---

## 7. 实战：Triton GEMM 的调优顺序

```
1. 先让 naive 版跑通                        ← 验证正确性
2. 加 shared memory tiling                  ← ~2-3× 提升
3. 加 autotune (BLOCK_M, BLOCK_N, BLOCK_K)  ← ~1.5-2× 提升
4. 调 num_warps                            ← ~1.1-1.3× 提升
5. 调 num_stages=2                         ← ~1.2-1.5× 提升（如果 memory bound）
6. 加 GROUP_M swizzling                    ← ~1.05-1.15× 提升
7. 考虑 split-K (if M/N small, K large)    ← 有场景依赖
8. 微调 BLOCK 尺寸到 GPU 特性              ← ~1.05-1.1× 提升

预期最终性能: 70-80% cuBLAS ← 对 Triton 来说已经很好
```

---

## 8. 参考资料

- [Triton MatMul Tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
- [How to Optimize a CUDA Matmul Kernel (Simon Boehm)](https://siboehm.com/articles/22/CUDA-MMM)
- [CUTLASS Documentation — GEMM](https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md)
- [NVIDIA Tensor Core Programming](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)
