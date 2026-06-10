# 11 — CUDA → Triton 迁移指南

> 如果你之前写 CUDA，或者是看 CUDA 教程学 GPU 的，这篇帮你快速把知识迁移到 Triton。

---

## 1. 心智模型转换

```
CUDA:
  "我有 N 个线程。每个线程应该处理哪个数据？怎么避免 bank conflict？
   怎么让 32 个线程的内存访问合并？"

Triton:
  "我有一个 block。block 应该处理多少数据？编译器帮我管线程。
   我应该用什么 BLOCK_SIZE？多少 num_warps？"
```

**关键洞察**: CUDA 是自底向上的（thread → warp → block），Triton 是自顶向下的（block → 编译器 → thread）。

---

## 2. 概念速查映射

| CUDA | Triton | 备注 |
|------|--------|------|
| `blockIdx.x` | `tl.program_id(0)` | Block 索引 |
| `blockIdx.y` | `tl.program_id(1)` | 2D grid 的第二维 |
| `gridDim.x` | `tl.num_programs(0)` | Block 总数 |
| `blockDim.x` | 由 `num_warps × 32` 隐式决定 | Triton 不需要手动设置 |
| `threadIdx.x` | 编译器自动分配 | 隐藏在 layout encoding 中 |
| `__shared__ float tile[32][32]` | 编译器自动 staging | 通过 `tl.load` 隐式使用 |
| `__syncthreads()` | 编译器自动插入 | 只在必要时加 barrier |
| `__global__ void kernel(...)` | `@triton.jit def kernel(...)` | 函数声明 |
| `kernel<<<grid, block>>>(args)` | `kernel[grid](args)` | kernel launch |
| `float* ptr` | `x_ptr` (raw pointer) | 内存访问 |
| `atomicAdd(ptr, val)` | `tl.atomic_add(ptr, val)` | 原子操作 |

---

## 3. 最常见的 CUDA→Triton 翻译

### 3.1 Vector Add

```cuda
// CUDA
__global__ void add_kernel(float* x, float* y, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = x[idx] + y[idx];
    }
}

void add(float* x, float* y, int n) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    add_kernel<<<grid_size, block_size>>>(x, y, out, n);
}
```

```python
# Triton
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)

def add(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, out, n)
    return out
```

**翻译要点**:
- `blockDim.x * threadIdx.x` → `tl.arange(0, BLOCK_SIZE)` — 编译器自动分配
- `if (idx < n)` → `mask = offsets < n` — 向量化 mask
- `grid_size` → `lambda meta: (...)` — 支持 autotuner 动态调整

### 3.2 Tiled MatMul

```cuda
// CUDA (简化版)
__global__ void matmul_kernel(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    
    float sum = 0.0;
    for (int k = 0; k < K; k += BLOCK_SIZE) {
        // 协作加载 A tile 到 shared memory
        As[threadIdx.y][threadIdx.x] = A[row * K + (k + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(k + threadIdx.y) * N + col];
        __syncthreads();
        
        for (int i = 0; i < BLOCK_SIZE; i++) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        __syncthreads();
    }
    C[row * N + col] = sum;
}
```

```python
# Triton
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        
        a = tl.load(a_ptrs, mask=..., other=0.0)  # 自动 shared memory staging
        b = tl.load(b_ptrs, mask=..., other=0.0)
        acc += tl.dot(a, b)  # 自动 MMA
    
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=...)
```

**翻译要点**:
- `__shared__ As[32][32]` + `__syncthreads()` → `tl.load` 自动 staging — 消失的复杂性
- 嵌套的 K 循环 → `tl.dot` 自动 MMA — 一行代替多行
- `threadIdx.y/x` → `tl.arange(0, BLOCK_M/N)` — 声明式而非索引式

---

## 4. 什么在 Triton 中变简单了？

### 4.1 Shared Memory Staging — 完全自动化

```
CUDA: ~20 行
  - 声明 shared memory
  - 协作加载（thread 偏移计算）
  - __syncthreads 同步
  - 可能的 bank conflict padding

Triton: 1 行
  x = tl.load(ptr + offsets, mask=mask)
  // 编译器自动处理: coalescing, staging, sync, bank conflict avoidance
```

### 4.2 Coalescing — 编译器分析

```
CUDA: 需要手动确保相邻线程访问相邻地址
  - 选择 row-major / column-major
  - 可能重新调整 thread index 的映射
  - 测试不同 access pattern

Triton: 编译器通过 layout order 自动确保 coalescing
  - order=[0,1] → row-major access
  - 编译器分析内存访问模式，生成 coalesced load/store
```

### 4.3 Block Size Tuning — Autotuner

```
CUDA: 手动试
  - 改 BLOCK_SIZE → 重新编译 → 跑 benchmark → 记录 → 换下一个值
  - 不同 GPU 需要不同的值

Triton: Autotuner
  @triton.autotune(configs=[...], key=['N'])
  - 列出候选 → 第一次运行自动测试 → 缓存最优 → 后续直接用
```

---

## 5. 什么在 Triton 中变难了（或做不到）？

### 5.1 精细的线程控制

```
CUDA 能做到但 Triton 做不到:
  - 指定"线程 0 做这个，线程 1 做那个"的不对称操作
  - Warp shuffle 显式操作（__shfl_sync）
  - 手动 warp specialization
  - 精确的 shared memory bank 控制

Triton 的设计哲学: 这些细节不应该由程序员管。
但对于极致性能优化，这些是不可或缺的。
```

### 5.2 Debug

```
CUDA: cuda-gdb, NVIDIA Nsight, printf
Triton: TRITON_INTERPRET（解释执行）, tl.device_print, IR dump
→ Triton 的调试工具链还不够成熟（详见笔记 06）
```

### 5.3 特殊的硬件特性

```
Triton 不能直接使用:
  - TMA (H100 硬件拷贝)
  - wgmma (H100 warp group MMA)
  - Thread Block Cluster (H100 block 协作)
  - 2:4 sparsity 硬件加速

这些是为什么 cuBLAS 在 H100 上比 Triton 快 15-30%。
```

---

## 6. 迁移策略：从 CUDA 到 Triton 的实战步骤

```
1. 找到你要迁移的 CUDA kernel
2. 识别 kernel 的 grid 结构（1D? 2D? 3D?）
3. 画出数据流图（不涉及线程分配）
4. 用 Triton 重写:
   - grid = block 的分配（和 CUDA 一致）
   - offsets = tl.arange 代替 threadIdx 计算
   - tl.load/tl.store 代替手写 shared memory
   - tl.dot 代替手写 inner loop
5. 加 autotune
6. 对比性能（预期: 70-90% of cuBLAS）

常见陷阱:
  - Stride 顺序: CUDA 的行/列容易混淆
  - dtype: Triton 中 fp16 输入自动被 tl.load 读取
  - 指针: Triton 的 ptr + offset 和 CUDA 指针运算相同
```

---

## 7. 什么时候应该留在 CUDA？

```
以下场景建议继续用 CUDA/CUTILE 而不是迁移到 Triton:

1. 你的 kernel 需要 95%+ peak TFLOPS（H100 上）
   → Triton 没有 TMA/wgmma/warp specialization

2. 极其精细的线程控制
   → 如 warp-level 的 producer/consumer pipeline

3. 稀疏计算
   → Triton 的 sparsity 支持不成熟

4. 已有大量 CUDA 代码，迁移成本 > 收益
   → 但可以新增的 fusion kernel 用 Triton 写
```

---

## 8. 参考资料

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Triton Official Tutorials](https://triton-lang.org/main/getting-started/tutorials/)
