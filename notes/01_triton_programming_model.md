# 01 — Triton 编程模型

> **目标**: 读完这篇，你能写出第一个 Triton kernel 并理解每一行的含义。
> **前置**: 笔记 00（GPU 执行模型）

---

## 0. 为什么需要 Triton？— 从 CUDA 的痛点说起

### 0.1 手写 CUDA 有多痛苦

用 CUDA 写一个最简单的矩阵乘法，你需要处理：

```cuda
// CUDA: ~150 行，你需要手写所有这些
__global__ void matmul_kernel(float* A, float* B, float* C, int M, int N, int K) {
    // 1. 计算每个线程负责哪部分数据
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 2. 手动管理 shared memory
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    // 3. 手动做 tiling 循环
    // 4. 手动 __syncthreads() 同步
    // 5. 手动处理边界（masking）
    // 6. 手动优化 coalescing（合并访问）
    // 7. 手动选择 BLOCK_SIZE（试错）
    // ...
}
```

**核心问题**: CUDA 让你以 **thread 为中心**编程——你需要告诉每个线程做什么，但这恰恰是最难的部分：
- 32 个线程怎么协作加载数据？（coalescing）
- Shared memory 怎么分配才不会 bank conflict？
- Block 大小怎么选？（太小浪费，太大 occupancy 低）

这些问题的答案取决于 GPU 型号（V100 vs A100 vs H100 完全不同），也取决于输入尺寸。

### 0.2 Triton 的答案：提升抽象层级

```
        CUDA:  你管 thread，编译器帮你管不了什么
        Triton: 你管 block，编译器帮你管 thread 的所有细节

        类比:
        CUDA = 汇编语言（管每个寄存器）
        Triton = C 语言（管变量，编译器管寄存器分配）
```

Triton 的核心洞察：
- **你不需要管 thread**。你只需要描述"一个 block 处理多少数据"
- 编译器自动决定 thread→数据 的映射（coalescing、bank conflict avoidance）
- 编译器自动插入同步（barrier placement）
- Autotuner 自动帮你选最优的 block 大小

---

## 1. Triton 的核心抽象：Program = Block

```
CUDA 的层级:            Triton 的层级:
  Thread ← 你管理          Thread ← 编译器自动管理
  Warp   ← 你需要注意       Warp   ← 编译器自动管理
  Block  ← 你管理          Block  ← 你管理（Program）
  Grid   ← 你管理          Grid   ← 你管理
```

在 Triton 中，**一个 program = 一个 CUDA thread block**。你写的 kernel 代码描述了一个 block 的行为，Triton 编译器负责：
1. 把这个 block 级代码展开成 thread 级代码
2. 自动分配数据到各线程（layout encoding）
3. 自动插入 shared memory staging
4. 自动处理 coalescing 和 bank conflict

---

## 2. 你的第一个 Triton Kernel（逐行讲解）

```python
import triton
import triton.language as tl
import torch

@triton.jit                    # ← 标记: "这是 Triton kernel，请编译"
def vector_add_kernel(
    x_ptr,                     # 输入数组 A 的指针（在 GPU 内存中）
    y_ptr,                     # 输入数组 B 的指针
    out_ptr,                   # 输出数组的指针
    n_elements,                # 数组长度
    BLOCK_SIZE: tl.constexpr,  # ← [关键] 编译时常量：每个 block 处理多少元素
):
    """
    每个 element 的计算: out[i] = x[i] + y[i]
    这个 kernel 描述了"一个 block"的行为。
    """
    # ── Step 1: 我是哪个 block？ ──
    # tl.program_id(0) = 当前 block 在 grid 中的索引
    # 类似 CUDA 的 blockIdx.x
    pid = tl.program_id(axis=0)
    
    # ── Step 2: 我负责哪些数据？ ──
    # 计算这个 block 在全局数据中的起始位置
    block_start = pid * BLOCK_SIZE
    
    # tl.arange(0, N) 生成 [0, 1, 2, ..., N-1] 的向量
    # [COMPILER] 这会在编译时展开为常量向量，不是运行时循环
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # ── Step 3: 处理边界 ──
    # 如果 n_elements 不是 BLOCK_SIZE 的整数倍，
    # 最后一个 block 的尾部需要 mask 掉
    mask = offsets < n_elements
    
    # ── Step 4: 从 HBM 加载数据 ──
    # 编译器自动处理 coalescing（合并访问）
    # other=0.0: 被 mask 掉的元素（越界）取这个值
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # ── Step 5: 计算 ──
    # 逐元素加法——编译器会自动分配每个线程做哪些元素的加法
    output = x + y
    
    # ── Step 6: 写回 HBM ──
    tl.store(out_ptr + offsets, output, mask=mask)


# ── Python 包装函数 ──
# Triton kernel 不能直接被 Python 调用，需要包装函数来：
# 1. 分配输出内存
# 2. 计算 grid 大小
# 3. 启动 kernel
def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)       # 分配输出 tensor
    n_elements = x.numel()              # 总元素数
    
    # grid: "需要多少个 block？"
    # 每个 block 处理 BLOCK_SIZE 个元素，一共 N 个元素
    # triton.cdiv = ceil division（向上取整）
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    
    # 启动 kernel: kernel_name[grid](args...)
    # [grid] 语法是 Triton 的特殊语法，表示 "用这个 grid 启动"
    vector_add_kernel[grid](x, y, output, n_elements)
    return output
```

### 2.1 这段代码在 GPU 上实际发生了什么？

```
假设 n_elements=1000, BLOCK_SIZE=256:

GPU 上会启动 4 个 block:
  Block 0: 处理元素 0-255    (pid=0, block_start=0)
  Block 1: 处理元素 256-511  (pid=1, block_start=256)
  Block 2: 处理元素 512-767  (pid=2, block_start=512)
  Block 3: 处理元素 768-999  (pid=3, block_start=768, mask掉760-999)

如果 GPU 有 132 个 SM，这 4 个 block 会被分配到 4 个不同的 SM 上
并行执行——每个 SM 跑一个 block，同时加载、计算、写回。

Block 内部: 编译器自动把 256 个元素分配到该 block 的线程中。
如果 num_warps=4 (8 warps = 128 线程)，每个线程处理 2 个元素。
```

### 2.2 关键概念速查

| 概念 | Triton 代码 | 含义 | 类比 CUDA |
|------|-----------|------|-----------|
| Block 索引 | `tl.program_id(0)` | 当前是第几个 block | `blockIdx.x` |
| Block 总数 | `tl.num_programs(0)` | 一共有多少个 block | `gridDim.x` |
| 向量生成 | `tl.arange(0, N)` | 生成 [0, 1, ..., N-1] | `threadIdx.x` (但自动展开) |
| 加载 | `tl.load(ptr, mask, other)` | 从 HBM 读数据 | 自动 coalesced ld.global |
| 存储 | `tl.store(ptr, val, mask)` | 写数据到 HBM | 自动 coalesced st.global |
| 矩阵乘 | `tl.dot(a, b)` | MMA 运算 | `mma.sync` (PTX) |
| 常量 | `tl.constexpr` | 编译时常量 | C++ template param |

---

## 3. 编译时常量 `tl.constexpr`

```python
def kernel(..., BLOCK_SIZE: tl.constexpr):
    # BLOCK_SIZE 必须在编译时已知
    # [COMPILER] 编译器会为每个不同的 BLOCK_SIZE 值生成独立版本
    # 类比: C++ template<BLOCK_SIZE> 或 MLIR constant attribute

# 为什么需要 constexpr？
# GPU 上很多优化依赖编译时已知的常数:
#   - 循环展开: for i in range(BLOCK_SIZE) → 完全展开
#   - 寄存器分配: 已知数组大小 → 可以全放寄存器
#   - 指令选择: 不同 BLOCK_SIZE → 不同 MMA 指令
```

---

## 4. Autotuning：让编译器帮你选最优参数

写 Triton kernel 时最头疼的问题：**block 大小选多少？**

- `BLOCK_SIZE=64`: block 太小，SM 利用率低
- `BLOCK_SIZE=256`: 不错
- `BLOCK_SIZE=2048`: block 太大，寄存器不够，occupancy 降低
- 最优值取决于：GPU 型号、数据大小、dtype...

Triton 的做法：你提供**候选配置列表**，它帮你找到最优的。

```python
@triton.autotune(
    configs=[
        # 候选配置 1
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        # 候选配置 2
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        # 候选配置 3
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        # 候选配置 4
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['n_elements'],  # 按输入大小选择——大小不同，最优配置可能不同
)
@triton.jit
def my_kernel(...):
    ...

# [COMPILER] 工作流程:
# 1. 第一次调用: 对每个 config 生成一个编译版本（JIT compile）
# 2. 在 GPU 上实际运行每个版本，测时间
# 3. 缓存最优结果到 ~/.triton/cache/
# 4. 后续调用: 直接使用缓存的最优版本
# 
# 类比: profile-guided optimization (PGO) + multi-versioning
```

### 4.1 Autotune 参数的物理意义

| 参数 | 含义 | 调大 | 调小 |
|------|------|------|------|
| `BLOCK_SIZE` | 每个 block 处理的元素数 | 更高利用率，但可能 occupancy 下降 | 更多 block，调度更灵活 |
| `num_warps` | 每个 block 的 warp 数 | 更多并发，更好的延迟隐藏 | 每 warp 有更多寄存器 |
| `num_stages` | Software pipeline 级数 | 更好的计算/加载重叠 | 更少 shared memory 占用 |

> 💡 **实战建议**: 开始时从现有 kernel 的 autotune 配置抄一份，跑起来后再根据自己的 GPU 和数据调整。

---

## 5. Grid 的计算

```python
# 1D grid（vector add, softmax, layernorm — 每个 block 处理一行或一段）
grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
#   ↑ lambda meta: 让 autotuner 传入当前测试的 config

# 2D grid（matmul — 每个 block 处理 C 的一块 tile）
grid = lambda meta: (
    triton.cdiv(M, meta["BLOCK_M"]),  # M 维需要多少个 block
    triton.cdiv(N, meta["BLOCK_N"]),  # N 维需要多少个 block
)

# triton.cdiv(a, b) = ceil(a / b)
# 例: triton.cdiv(1000, 256) = 4
#         1000 / 256 = 3.90625 → ceil → 4
```

---

## 6. Triton vs CUDA 完整概念映射

| Triton | CUDA | 说明 |
|--------|------|------|
| `tl.program_id(0)` | `blockIdx.x` | block 索引 |
| `tl.program_id(1)` | `blockIdx.y` | 2D grid 的第二维 |
| `tl.num_programs(0)` | `gridDim.x` | block 总数 |
| `tl.arange(0, N)` | `threadIdx.x`（展开） | 编译器自动分配 thread |
| `tl.load` / `tl.store` | 自动 coalesced `ld.global` / `st.global` | 编译器处理合并访问 |
| `tl.dot` | `mma.sync` (PTX) | 自动映射到 Tensor Core |
| `tl.sum` / `tl.max` | warp shuffle + shared memory reduction | 编译器生成 reduction 代码 |
| `@triton.autotune` | 手选 block size + 试错 | 编译器帮你试 |
| `num_warps` | `blockDim.x / 32` | 控制每 block 的 warp 数 |
| `num_stages` | 手动 software pipeline | 编译器自动生成 pipeline 代码 |

---

## 7. 从零写一个 Triton kernel 的 Checklist

```
□ 1. 确定 grid 维度 (1D? 2D? 3D?)
     → 看你的数据自然是什么形状的

□ 2. 写 kernel 函数
     → tl.program_id 获取 block 索引
     → tl.arange 生成线程偏移
     → tl.load 加载数据（记得 mask 和 other）
     → 计算（使用 Triton language ops）
     → tl.store 写回数据

□ 3. 写 Python 包装函数
     → 分配输出 tensor
     → 计算 grid = lambda meta: (...)
     → kernel[grid](args...)

□ 4. 加 autotune（可选但推荐）
     → 列出候选 BLOCK_SIZE / num_warps / num_stages
     → 确定 key（按什么特征选配置）

□ 5. 测试正确性
     → 对比 PyTorch reference，max_diff < 1e-3

□ 6. Benchmark
     → CUDA Event 计时，计算 TFLOPS/带宽
     → 分析是 compute bound 还是 memory bound
```

---

## 参考资料

- [Triton 官方文档](https://triton-lang.org/)
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/)
- [Triton 论文 (Tillet et al., 2019)](https://dl.acm.org/doi/10.1145/3315508.3329973)
