# 02 — GPU 内存层级与优化

> **目标**: 理解 GPU 各层内存的特性，掌握 shared memory、coalescing、bank conflict 和 software pipelining 的优化原理。
> **前置**: 笔记 00（GPU 执行模型）、笔记 01（Triton 编程模型）

---

## 0. 先从直观感受开始：数据的旅程

```
你写:  x = tl.load(ptr + offsets)
你想:  从内存读一个数，简单

GPU 实际做的:
  ptr 指向哪里？
  ├── HBM (global memory):  走 ~500 cycles 到 L2 cache → 再到 SM
  ├── L2 Cache 命中:        走 ~200 cycles
  ├── Shared Memory:        走 ~20 cycles
  └── Register:             走 ~0 cycles（数据已经在寄存器里）

  这 500x 的速度差异，就是为什么"把数据放在哪里"决定了 kernel 性能。
```

**类比**: 你在图书馆查资料
- **寄存器** = 你桌上摊开的书，眼睛扫一眼就到
- **Shared Memory** = 同组同学桌上的书，站起来就能看到
- **L2 Cache** = 图书馆书架上的书，需要走过去拿
- **HBM** = 隔壁城市图书馆的书，要走很远但一次能搬回来很多

好的 kernel = 把要反复用到的数据提前搬到桌上来（shared memory staging），减少跑图书馆的次数。

---

## 1. 各层内存详细特性

| 特性 | HBM (Global) | L2 Cache | Shared Memory | Register |
|------|-------------|----------|---------------|----------|
| 容量 (H100) | 80 GB | 50 MB | 228 KB/SM | 256 KB/SM |
| 延迟 (近似) | 300-800 cycles | ~200 cycles | 20-30 cycles | ~0 |
| 带宽 | 3.35 TB/s | ~10 TB/s | ~128 B/clk/SM | — |
| 作用域 | 全 GPU | 全 GPU | Block 内 | 单线程 |
| 管理方式 | 你显式管理 | 硬件自动 | 编译器管理 | 编译器分配 |
| 变量在哪 | `torch.tensor` 的数据 | 自动缓存 | `tl.load` 加载后自动 staging | kernel 内局部变量 |

### 1.1 各层的带宽意味着什么？

```
HBM (3.35 TB/s):
  每秒可以读 3.35 × 10^12 字节
  = 每秒约 8.38 亿个 fp32 数字
  = 每毫秒约 83.8 万个

Shared Memory (~128 B/clk/SM, 132 SM @ 1.9 GHz):
  每秒约 128 × 1.9G × 132 ≈ 32 TB/s (总计，所有 SM)
  约 HBM 的 10 倍带宽

Register:
  每个 cycle 每个线程可以读 ~1 个寄存器
  一个 SM 有 2048 个活跃线程 × 1.9 GHz ≈ 3.9 T 次访问/秒
  但每次访问只读 4 字节（单精度）
```

---

## 2. Global Memory Coalescing（合并访问）— 最重要的一条优化

### 2.1 什么是合并访问？

同一个 warp 的 32 个线程访问 HBM 时，GPU 硬件会把它们的请求**合并**成尽可能少的内存事务（transaction）。

```
✅ 合并访问（coalesced）:
  线程 0 访问地址 0x100 → ┐
  线程 1 访问地址 0x104 →  ├── 1 次 128-byte transaction 全搞定
  线程 2 访问地址 0x108 →  │    (32 threads × 4 bytes = 128 bytes)
  ...                      │
  线程 31 访问地址 0x17C → ┘

❌ 非合并访问（strided）:
  线程 0 访问地址 0x100 →  ─── 1 次 transaction
  线程 1 访问地址 0x200 →  ─── 另 1 次 transaction (跳过 256 bytes)
  线程 2 访问地址 0x300 →  ─── 又 1 次 transaction
  ...
  最坏情况: 32 次独立的 transaction (32x 带宽浪费！)
```

### 2.2 什么导致非合并访问？

```python
# ✅ 合并访问: 相邻线程访问相邻内存地址
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
# thread 0 → offsets[0]; thread 1 → offsets[1]
# → 地址连续 → coalesced!

# ❌ 非合并访问: 跨行访问（strided access）
# 访问矩阵的一列时，相邻元素在内存中相距 1 行
offsets = tl.arange(0, N_ROWS) * N_COLS + col
# thread 0 → row 0; thread 1 → row 1
# → 地址相差 N_COLS × 4 bytes → 非合并！
```

### 2.3 Triton 中如何控制合并访问？

通过 **layout order** 参数：

```python
# order=[0,1]: dim 0 为 innermost（连续）
# → 线程沿 dim 1 方向连续 → 合并访问沿 dim 1
ptr = tl.make_block_ptr(
    base, shape, strides, offsets, block_shape,
    order=[0, 1]  # row-major: 线程沿列维连续
)

# order=[1,0]: dim 1 为 innermost
# → 线程沿 dim 0 方向连续 → 合并访问沿 dim 0
ptr = tl.make_block_ptr(
    base, shape, strides, offsets, block_shape,
    order=[1, 0]  # column-major
)
```

**经验法则**: 默认用 `order=[0,1]`（row-major，适合 PyTorch 默认的内存布局）。

> 🔧 **Compiler Perspective**: Coalescing 分析类似编译器的 stride 分析——检查相邻线程的地址差是否为 1（以元素为单位）。Triton 编译器在生成 PTX 时，会根据 layout encoding 自动生成 coalesced 的 `ld.global` 指令。

---

## 3. Shared Memory Bank Conflict

### 3.1 Bank 是什么？

Shared memory 被分成 32 个 bank，每个 bank 4 字节宽。这是硬件限制——每个 cycle 每个 bank 只能服务一个线程。

```
Shared Memory 的物理结构（简化）:
  Bank 0:  [addr 0] [addr 128] [addr 256] ... (每 128 bytes 一个 bank)
  Bank 1:  [addr 4] [addr 132] [addr 260] ...
  Bank 2:  [addr 8] [addr 136] [addr 264] ...
  ...
  Bank 31: [addr 124][addr 252] [addr 380] ...

  地址 % 128 / 4 = bank 编号（在 32-bank 模式下）
```

### 3.2 Bank Conflict 示例

```
✅ 无冲突: 32 个线程各访问不同 bank
  Thread 0 → addr 0   (bank 0)
  Thread 1 → addr 4   (bank 1)
  ...
  Thread 31 → addr 124 (bank 31)
  → 1 cycle 完成

❌ 2-way conflict: 两个线程访问同一个 bank
  Thread 0 → addr 0   (bank 0)
  Thread 1 → addr 128 (bank 0) ← 冲突！
  Thread 2 → addr 8   (bank 2)
  ...
  → 2 cycles（串行化）

❌ 32-way conflict: 所有线程访问同一个 bank
  Thread 0 → addr 0   (bank 0)
  Thread 1 → addr 128 (bank 0)
  ...
  Thread 31 → addr 3968 (bank 0)
  → 32 cycles！等价于没有 shared memory 的加速
```

### 3.3 如何避免 Bank Conflict？

**方法 1: 加 padding**

```python
# 分配 shared memory 时多加几个元素
# 把地址偏移一个 bank，让后续访问不再冲突
# 例: 原本每行 128 个 float → 改成 128 + 2 = 130
# 这样每行的 bank 分配就会错开

# Triton 编译器会自动做 swizzling，通常不需要手动 padding
# 但理解原理有助于 debug
```

**方法 2: 让 Triton 自动处理（大多数情况）**

Triton 编译器在生成 shared memory layout 时会自动应用**地址 swizzling**（地址重映射），减少 bank conflict。你通常不需要手动处理。

> 🔧 **Compiler Perspective**: Bank conflict 类似 CPU cache 的 structural hazard——多个请求竞争同一硬件资源。但与 CPU cache 不同，shared memory 是显式管理的，编译器可以在编译时分析访问模式并做静态优化（padding、swizzle pattern）。

---

## 4. Software Pipelining（软件流水线）

### 4.1 问题：计算和加载的串行瓶颈

```
最朴素的做法 (num_stages=1):
  ┌──────┬──────┬──────┬──────┐
  │Load 1│Comp 1│Load 2│Comp 2│  ← 加载和计算不重叠
  └──────┴──────┴──────┴──────┘
  Load 时计算单元空闲，Compute 时内存单元空闲
```

### 4.2 解决方案：Double Buffering

```
Double buffering (num_stages=2):
  ┌──────┬──────┬──────┐
  │Load 1│Load 2│Load 3│
  │      │Comp 1│Comp 2│  ← 加载和计算重叠！
  └──────┴──────┴──────┘
  
  原理: 两个 buffer（stage 0, stage 1）
  - Comp 1 使用 stage 0 时，Load 2 同时往 stage 1 写
  - Comp 2 使用 stage 1 时，Load 3 同时往 stage 0 写
```

### 4.3 Triton 中的使用

```python
@triton.autotune(
    configs=[
        triton.Config({...}, num_stages=2),  # double buffering
        triton.Config({...}, num_stages=3),  # 三级流水
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(...):
    # 你不需要在代码中做任何特殊处理！
    # Triton 编译器自动:
    #   1. 展开 K 维循环
    #   2. 插入 cp.async（异步拷贝）指令
    #   3. 插入 prefetch
    #   4. 管理 buffer 切换

# [COMPILER] 编译器的做法:
# - num_stages=2: 生成 double-buffered 循环，使用 cp.async 做异步加载
# - num_stages=3: 三级流水，更深的延迟隐藏
# - 这本质上就是 VLIW 编译器中的 modulo scheduling:
#   展开循环 → 重排指令 → 插入 prefetch → 计算 initiation interval
```

### 4.4 num_stages 选择的 tradeoff

```
num_stages=1:
  优点: 不占用额外的 shared memory
  缺点: 加载和计算串行，SM 利用率低
  适用: shared memory 已经很紧张的情况

num_stages=2 (最常用):
  优点: 加载和计算可以重叠，简单有效
  缺点: 需要 2x shared memory
  适用: 大多数 GEMM kernel

num_stages=3:
  优点: 更深的延迟隐藏（如果 HBM 延迟是瓶颈）
  缺点: 需要 3x shared memory
  适用: HBM 延迟特别大的场景（或数据量极大时）
```

---

## 5. Async Copy：硬件加速的数据搬运

GPU 的硬件在不断进化，提供了越来越强的**异步数据搬运**能力。

### 5.1 cp.async（Ampere A100+）

```
传统 load: 占用线程的寄存器，线程不能做其他事
  Thread: [等待 load 完成...] → [使用数据]

cp.async: 不占用线程寄存器，线程可以继续计算
  Thread: [发起 cp.async] → [做其他计算] → [等待完成] → [使用数据]
  硬件在后台偷偷把数据从 HBM 搬到 shared memory
```

Triton 在 `num_stages > 1` 时自动使用 `cp.async`，你不需要写任何特殊代码。

### 5.2 TMA（Tensor Memory Accelerator, Hopper H100）

H100 引入了 TMA——一个专门的**硬件数据搬运单元**：

```
cp.async (A100): 每个 warp 自己发起异步拷贝
TMA (H100):      专门的硬件单元做数据搬运，完全不占用计算资源
                 支持 2D tile copy（一次搬一个矩形区域）
                 自动做边界处理、地址计算

Triton 的状态:  目前（3.x）对 TMA 的支持有限
                这是为什么 H100 上 Triton 的 GEMM 不如手写 CUDA 的原因之一
```

---

## 6. 性能分析速查：怎么判断你的 kernel 瓶颈在哪？

### 6.1 快速判断

```
问自己 3 个问题:

1. 算术强度高吗？
   AI = 算法 FLOPs / HBM 访问字节数
   AI > GPU ridge point → compute bound
   AI < GPU ridge point → memory bound
   (ridge point: H100 ≈ 295 FLOP/byte, A100 ≈ 156 FLOP/byte)

2. 如果我减少 HBM 访问（加 fusion），性能能提升吗？
   → 如果能 → memory bound
   → 如果没用 → compute bound

3. 用 ncu 看 achieved occupancy
   → occupancy < 50% → 可能是 register/spared memory 压力太大
   → occupancy > 80% → 资源利用率不错
```

### 6.2 Kernel 类型的典型特征

| Kernel | 算术强度 | 瓶颈 | 优化方向 |
|--------|---------|------|---------|
| Elementwise (ReLU, add) | 0.1-0.5 | Memory | Fusion 减少 HBM round-trip |
| Reduction (softmax) | 1-5 | Memory | 减少 pass 数 |
| Small GEMM (<512) | 50-200 | 混合 | Shared memory tiling |
| Large GEMM (>4096) | 500+ | Compute | Tensor Core 利用率 |
| Flash Attention | 50-200 | Memory → Compute | Tiling 减少 HBM 流量 |

---

## 7. 总结：GPU 内存优化的思维框架

```
                      GPU 内存优化的四个层级

Level 1: 减少数据量
  → 用更小的 dtype (fp16 代替 fp32)
  → 剪枝/稀疏化

Level 2: 减少 HBM 访问次数
  → Operator fusion（合并多个 kernel 为一个）
  → Tiling（把数据搬到 shared memory，复用）

Level 3: 加速每次 HBM 访问
  → Coalescing（合并访问，减少 transaction 数）
  → Alignment（对齐到 128-byte 边界）

Level 4: 隐藏 HBM 延迟
  → Software pipelining（num_stages）
  → Async copy（cp.async / TMA）
  → Occupancy（足够多的 warp 等待时切换）
```

---

## 参考资料

- [CUDA C++ Best Practices Guide — Memory Optimizations](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations)
- [NVIDIA H100 Whitepaper — Memory Hierarchy](https://resources.nvidia.com/en-us-tensor-core)
- [Triton Language Reference — Load/Store](https://triton-lang.org/main/python-api/triton.language.html#loading-and-storing)
