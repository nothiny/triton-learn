# 00 — GPU 执行模型

> **目标读者**: 从零开始。即使你从来没写过 GPU 代码，这篇笔记也能帮你建立完整的心智模型。
> 如果你有编译器后端经验，重点关注 🔧 标记的段落。

---

## 0. 先建立直觉：为什么需要 GPU？

### 从一道小学数学题开始

假设计算 `C = A + B`，A 和 B 各有一百万个数字：

**CPU 的做法（一个数学天才）:**
按顺序一个个加：第1个、第2个、第3个... 共一百万步
每个加法 0.2 纳秒，总共 0.2 毫秒 — 很快！

**GPU 的做法（一万个小学生）:**
每人领 100 个数字，同时开始加
单个学生可能慢一些（每个加法 2 纳秒），但一万人同时工作
总时间 ≈ 2 纳秒 × 100 = 0.2 微秒 — 快了 1000 倍！

这个类比抓住了 GPU 的本质：

| | CPU（天才） | GPU（小学生军团） |
|---|---|---|
| 单个任务速度 | **极快** | 较慢 |
| 同时做几件事 | 几个（4-64 核） | **几千个** |
| 擅长 | 复杂逻辑、分支、串行 | 简单重复、大量数据 |
| 设计哲学 | **Latency-oriented**（降低单次延迟） | **Throughput-oriented**（提高总吞吐） |

> 🔧 **Compiler Perspective**: CPU 优化关注减少关键路径延迟（instruction scheduling, bypass network）。GPU 优化关注提高吞吐（occupancy, memory coalescing）。两者用的技术完全不同。

### GPU 长什么样？

```
┌─────────────────────────────────────────────────────────┐
│                    NVIDIA GPU (如 H100)                  │
│  ┌───────────────────────────────────────────────────┐  │
│  │                    HBM (80 GB)                     │  │
│  │              显卡的内存，类似电脑的 RAM              │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐     ┌──────┐     │
│  │ SM 0 │ │ SM 1 │ │ SM 2 │ │ SM 3 │ ... │SM 131│     │
│  │ ┌──┐ │ │ ┌──┐ │ │ ┌──┐ │ │ ┌──┐ │     │ ┌──┐ │     │
│  │ │小│ │ │ │小│ │ │ │小│ │ │ │小│ │     │ │小│ │     │
│  │ │学│ │ │ │学│ │ │ │学│ │ │ │学│ │     │ │学│ │     │
│  │ │生│ │ │ │生│ │ │ │生│ │ │ │生│ │     │ │生│ │     │
│  │ │军│ │ │ │军│ │ │ │军│ │ │ │军│ │     │ │军│ │     │
│  │ │团│ │ │ │团│ │ │ │团│ │ │ │团│ │     │ │团│ │     │
│  │ └──┘ │ │ └──┘ │ │ └──┘ │ │ └──┘ │     │ └──┘ │     │
│  └──────┘ └──────┘ └──────┘ └──────┘     └──────┘     │
│                                                         │
│  H100: 132 个 SM，每个 SM 可以同时跑几千个线程           │
└─────────────────────────────────────────────────────────┘
```

- **SM (Streaming Multiprocessor)** = GPU 的计算核心，每个 SM 独立运行
- **HBM (High Bandwidth Memory)** = GPU 的"大内存"，所有 SM 共享
- H100 有 **132 个 SM**，每个 SM 能同时处理 **2048 个线程**

---

## 1. 线程层级：Grid → Block → Warp → Thread

GPU 的线程不是扁平的，而是分层的。理解这个层级是理解 GPU 编程的第一步。

```
一个 Kernel（你写的函数）
  │
  └── Grid（整个计算任务）
        │
        ├── Thread Block 0 ── 映射到 SM 0
        │     ├── Warp 0 (32 个线程，锁步执行)
        │     │     ├── Thread 0
        │     │     ├── Thread 1
        │     │     └── ... (共 32 个)
        │     ├── Warp 1 (另外 32 个线程)
        │     └── ... (最多 64 个 warp/block)
        │
        ├── Thread Block 1 ── 映射到 SM 1
        └── ...
```

**逐层解释：**

### Thread（线程）— 最小执行单元
- 每个线程执行相同的 kernel 代码
- 不同线程通过 `threadId` 区分自己该处理哪部分数据
- 类比：一个小学生

### Warp（线程束）— 调度的最小单位
- **32 个线程组成一个 warp**，所有线程**同时执行同一条指令**
- 这是 GPU 最核心的约束：一个 warp 内的 32 个线程**必须**在同一时刻执行相同的指令
- 类比：一个老师同时指挥 32 个学生做同样的动作
- 如果一个 warp 内的线程走了不同的分支（if/else），两边的代码都会被执行，只是部分线程的结果被丢弃——这叫 **warp divergence**，后面会详细讲

### Thread Block（线程块）— 协作单元
- 多个 warp 组成一个 block（通常 128-1024 个线程）
- **同一个 block 内的线程可以**：
  - 通过 **shared memory**（共享内存）快速交换数据
  - 通过 `__syncthreads()` / `bar.sync` 做同步
- **不同 block 之间不能直接通信**（只能通过 global memory，很慢）
- 一个 block 被分配到一个 SM 上执行

### Grid（网格）— 整个 kernel
- 所有 block 的集合
- Grid 的大小 = 你的数据大小 / 每个 block 处理的数据量

### 为什么分这么多层？

**如果不分层（1 万个独立线程）:**
- 线程间无法协作（不能共享中间结果）
- 硬件调度开销大（每个线程都要单独调度）

**分成 Block → Warp:**
- Block 内的线程通过 shared memory 协作
- 硬件以 warp（32 线程）为单位调度，效率高
- Block 可以分配到不同的 SM，充分利用硬件

> 🔧 **Compiler Perspective**: Grid/Block/Warp 的层级关系类似于嵌套并行循环的 tiling 分解：
> - Grid = 最外层并行循环
> - Block = tile（分块）
> - Warp = 向量化宽度（SIMD width）
> - 编译器在 Triton 中自动做这个分解，你只需要指定 block 大小。

---

## 2. SIMT 执行模型：为什么"32 个线程必须执行同一条指令"

这是理解 GPU 最关键的概念。

### 2.1 SIMD vs SIMT

CPU 也有"同时处理多个数据"的能力（SIMD: AVX, SSE），但 GPU 的做法不同：

| | SIMD (CPU, 如 AVX-512) | SIMT (GPU) |
|---|---|---|
| 宽度 | 固定硬件宽度（512 bit = 16×32b） | 逻辑上 32 threads/warp |
| 写代码 | 你写 `_mm512_add_ps()`，显式用向量 | 你写标量代码，编译器映射到 warp |
| 分支 | 不支持（如果有 if，CPU 标量执行） | 支持但代价：两路都执行 |
| 同步 | 不需要 | warp 内隐式同步 |

**核心区别**: 在 GPU 上你写的是**标量代码**（像给一个线程写的），但硬件以 warp 为单位并行执行。这叫 **SIMT (Single Instruction, Multiple Threads)**。

### 2.2 Warp Divergence — GPU 编程最大的性能陷阱

```cuda
// ❌ 不好的代码：warp 内分支
if (threadIdx.x < 16) {
    // 16 个线程走这条路
    x = expensive_op_a(x);  // 花费 100 cycles
} else {
    // 另外 16 个线程走这条路
    x = expensive_op_b(x);  // 花费 100 cycles
}
// 实际执行时间 = 200 cycles！
// 为什么？因为 32 个线程锁步执行：
//   cycles   0-100: threads 0-15 执行 op_a，threads 16-31 空转（masked off）
//   cycles 100-200: threads 16-31 执行 op_b，threads 0-15 空转
```

```
可视化:

时间 →
Warp 内的 32 个线程:
  Thread  0: [══════ op_a ══════][         空转          ]
  Thread  1: [══════ op_a ══════][         空转          ]
  ...
  Thread 15: [══════ op_a ══════][         空转          ]
  Thread 16: [       空转       ][══════ op_b ══════]
  ...
  Thread 31: [       空转       ][══════ op_b ══════]
  
  总时间 = op_a 的时间 + op_b 的时间（而不是 max(op_a, op_b)）
```

```cuda
// ✅ 好的代码：让 warp 内所有线程做同样的事
// 如果分支不可避免，确保同一 warp 内走同一分支
if (warpId < 16) {
    x = expensive_op_a(x);  // warp 0-15: 100 cycles
} else {
    x = expensive_op_b(x);  // warp 16-31: 100 cycles
}
// 总时间 = 100 cycles（因为不同 warp 可以独立执行）
```

**经验法则**: 避免让 warp 内的线程走不同分支。如果必须有分支，让分支边界对齐 warp 边界（32 的倍数）。

> 🔧 **Compiler Perspective**: Divergence 处理类似于编译器的 if-conversion + predication。GPU 编译器会计算 divergence 代价，代价低时用 predication（两路都执行但用 mask 选结果），代价高时才用真正的分支 + reconverge point。Triton 编译器会自动做这个决策。

---

## 3. 内存层级：从快到慢，从小到大

GPU 的内存系统是分层的——越快的越小，越大的越慢。

```
┌──────────────────────────────────────────────────────┐
│                    Register File                      │
│  每 SM: 256 KB (65536 × 32-bit)                      │
│  速度: ~0 cycles（即刻访问）                          │
│  作用域: 单个线程                                     │
│  类比: 你桌上的草稿纸 — 伸手就能拿到                   │
├──────────────────────────────────────────────────────┤
│                   Shared Memory                       │
│  每 SM: 228 KB (H100)                                │
│  速度: ~20-30 cycles                                 │
│  作用域: Block 内所有线程共享                          │
│  类比: 你小组共用的白板 — 站起来就能写                  │
├──────────────────────────────────────────────────────┤
│                  L1 / L2 Cache                        │
│  L1: 256 KB/SM, L2: 50 MB (H100)                    │
│  速度: ~200-300 cycles                               │
│  作用域: SM 内 / 全 GPU                               │
│  类比: 教室里的书架                                    │
├──────────────────────────────────────────────────────┤
│                HBM (Global Memory)                    │
│  容量: 80 GB (H100)                                  │
│  速度: ~300-800 cycles                               │
│  带宽: H100 3.35 TB/s                                │
│  作用域: 全 GPU + 可通过 PCIe 与 CPU 通信             │
│  类比: 学校图书馆 — 要走很远，但一次能搬很多书           │
└──────────────────────────────────────────────────────┘
```

### 3.1 速度差异到底有多大？

用人类能感知的时间缩放（1 cycle = 1 秒）来感受：

| 操作 | GPU cycles | 人类时间 |
|------|-----------|---------|
| 寄存器访问 | ~0 | 伸手拿笔 |
| Shared memory | ~20-30 | 站起来写白板（半分钟） |
| L1 hit | ~200 | 去教室书架（3 分钟） |
| L2 hit | ~300 | 去图书馆（5 分钟） |
| HBM | ~300-800 | 去隔壁城市（5-13 分钟） |
| CPU→GPU (PCIe) | ~10,000+ | 去月球 |

**核心启示**: 好的 kernel 会尽量把数据放在离计算单元近的地方（寄存器、shared memory），减少远距离数据搬运。

### 3.2 各层级的详细特性

| 层级 | 容量 (H100/SM) | 延迟 | 带宽 | 作用域 | 管理方式 |
|------|----------------|------|------|--------|----------|
| Register | 65536×32b/SM | ~0 cycles | — | 单线程 | 编译器分配 |
| Shared Memory | 228 KB/SM | ~20-30 cycles | ~128 B/cycle/SM | Block 内 | 程序员/编译器管理 |
| L1 Cache | 256 KB/SM | ~200 cycles | — | SM 内 | 硬件自动 |
| L2 Cache | 50 MB | ~300 cycles | — | 全 GPU | 硬件自动 |
| HBM | 80 GB | ~300-800 cycles | 3.35 TB/s (H100) | 全 GPU + CPU | 显式管理 |

> 🔧 **Compiler Perspective**: Shared memory 类似嵌入式 DSP 的 scratchpad memory（显式管理的片上 SRAM），不同于 CPU 的透明 cache。寄存器分配在 GPU 上不再只是"减少 spill → 更快"，而是"少用寄存器 → 更多 warp → 更好的延迟隐藏"——这是一个多目标优化问题。

---

## 4. Occupancy（占用率）与延迟隐藏

### 4.1 为什么 GPU 需要同时跑这么多线程？

GPU 的核心秘密——**用并行度换延迟**。

```
场景: 一个 warp 发出 HBM 读取请求

CPU 的做法:
  线程 1: 读内存... 等 300 cycles... 数据到了，继续算
  等待期间 CPU 可以切到另一个线程（context switch，开销 ~100 cycles）

GPU 的做法:
  Warp 0: 读内存 ───── 等 300 cycles ───── 数据到了，继续算
  Warp 1:       读内存 ───── 等 300 cycles ───── 数据到了
  Warp 2:              读内存 ───── 等 300 cycles ─────
  ...
  Warp 16:                                  读内存...
  
  SM 只要在 300 cycles 内不断有 warp 可以切换（~0 cycle 切换开销！），
  计算单元就永远不会空闲。300 cycles / 32 cycles per warp = 约 10 个 warp 就够了。
```

### 4.2 Occupancy 公式

$$
\text{Occupancy} = \frac{\text{当前 SM 上活跃的 warp 数}}{\text{SM 最大 warp 数}}
$$

- **高 occupancy**：SM 上有足够多的 warp，能有效隐藏内存延迟
- **低 occupancy**：SM 上 warp 太少，计算单元经常空闲等数据

### 4.3 什么限制 occupancy？

一个 SM 的物理资源是固定的，资源被占用越多，能放的 warp 就越少：

```
每个 SM 的资源预算 (H100):
  ┌──────────────────┬─────────────┐
  │ 资源              │ 总量        │
  ├──────────────────┼─────────────┤
  │ 寄存器            │ 65536 × 32b │
  │ Shared Memory     │ 228 KB      │
  │ 最大 Warp 数      │ 64 (2048 threads) │
  │ 最大 Block 数     │ 32          │
  └──────────────────┴─────────────┘
```

示例计算:

每线程用 255 个寄存器，32 个线程每 warp:

$$
\begin{aligned}
255 \times 32 &= 8160 \text{ 寄存器/warp} \\
65536 / 8160 &\approx 8 \text{ 个 warp（最多）} \\
\text{Occupancy} &= 8 / 64 = 12.5\% \text{（很低！）}
\end{aligned}
$$

每线程用 64 个寄存器，32 个线程每 warp:

$$
\begin{aligned}
64 \times 32 &= 2048 \text{ 寄存器/warp} \\
65536 / 2048 &= 32 \text{ 个 warp} \\
\text{Occupancy} &= 32 / 64 = 50\% \text{（不错）}
\end{aligned}
$$

**核心 tradeoff**：每线程多用寄存器 → 能做更复杂的计算，但 occupancy 降低 → 延迟隐藏能力差。优秀的 GPU 程序员需要在这之间找到平衡点。

> 🔧 **Compiler Perspective**: GPU 的寄存器分配是一个三资源约束的优化问题：registers、shared memory、warps 相互竞争 SM 资源。这与传统 CPU 寄存器分配根本不同——在 CPU 上，spill 总是坏的；在 GPU 上，少用寄存器可能比避免 spill 更重要（因为可以提高 occupancy）。Triton 的 autotuner 本质上是在搜索这个资源空间。

---

## 5. Tensor Core：GPU 的"作弊器"

### 5.1 普通计算 vs Tensor Core

**普通 CUDA Core（FP32）:**

一次做 1 个乘法 + 1 个加法 = 1 FMA，需要 2 cycles

**Tensor Core（FP16）:**

$$
16 \times 8 \times 16 \text{ 的矩阵乘加} = 2048 \text{ 个 FMA，只需要 1 cycle}
$$

吞吐量是 CUDA Core 的 ~16 倍！

Tensor Core 是专门为矩阵乘法设计的硬件单元。它不灵活（只能做 MMA），但做 MMA 时极快。

### 5.2 MMA 指令的工作方式

`mma.sync.aligned.m16n8k16` 这条 PTX 指令的含义：

$$
m = 16, \quad n = 8, \quad k = 16
$$

$$
A[16 \times 16] \times B[16 \times 8] + C[16 \times 8] \to D[16 \times 8]
$$

一个 warp 的 32 个线程协作完成这个计算:

- 线程 0: 持有 A 的一部分 + B 的一部分 + C 的一部分
- 线程 1: 持有 A 的另一部分 + B 的另一部分 + C 的另一部分
- ...
- 线程 31: (同上)

计算后，每个线程持有结果矩阵 D 的不同"碎片"（fragment）

### 5.3 各代 Tensor Core 对比

| GPU | Tensor Core 代 | FP16 MMA | FP8 MMA | 特殊能力 |
|-----|---------------|----------|---------|---------|
| V100 | 1st Gen | m8n8k4 | — | — |
| A100 | 3rd Gen | m16n8k16 | — | sparsity (2x) |
| H100 | 4th Gen | m16n8k16 | m16n8k32 | wgmma, TMA |

H100 的 `wgmma`（warp group MMA）允许 4 个 warp（128 线程）协作做更大的 MMA，是 Triton 3.x 还无法直接使用的特性。

> 🔧 **Compiler Perspective**: MMA 指令类似于 SIMD intrinsic，但涉及 warp 内跨线程数据共享。Triton 的 `tl.dot` 自动映射到 MMA，编译器负责生成正确的 fragment 布局（哪个线程持有矩阵的哪部分）。这个映射规则就是 TTGIR 中的 `MmaEncodingAttr`。

---

## 6. 总结：GPU 编程的核心心智模型

```
                     GPU 编程的 3 个核心约束

1. SIMT 锁步执行
   ┌─────────────────────────────────────────────┐
   │ Warp 内 32 线程执行同一条指令                │
   │ 分支 → 两路都执行 → 2x 延迟                  │
   │ 解法: 让同一 warp 走同一分支                  │
   └─────────────────────────────────────────────┘

2. 内存层级（数据局部性）
   ┌─────────────────────────────────────────────┐
   │ 寄存器 (OD cycles) → Shared (20) → HBM (500) │
   │ 尽可能把数据留在离计算近的地方               │
   │ 解法: tiling + shared memory staging          │
   └─────────────────────────────────────────────┘

3. Occupancy（延迟隐藏）
   ┌─────────────────────────────────────────────┐
   │ 多 warp = 内存等待期间有活干                 │
   │ 寄存器/spared memory 用太多 → warp 减少       │
   │ 解法: 平衡资源使用                            │
   └─────────────────────────────────────────────┘
```

这三个约束贯穿后续所有 kernel 的设计决策。当你看到 Phase 2 的 tiled matmul 时，回想这里：
- **Tiling** = 应对内存层级（把数据搬到 shared memory）
- **num_warps** = 应对 occupancy（控制并发 warp 数）
- **避免 warp divergence** = 应对 SIMT 约束

---

## 参考资料

- [CUDA C++ Programming Guide — Hardware Implementation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#hardware-implementation)
- [NVIDIA H100 Architecture Whitepaper](https://resources.nvidia.com/en-us-tensor-core)
- [NVIDIA A100 Tensor Core Architecture](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf)
- [Triton 论文 (Tillet et al., 2019)](https://dl.acm.org/doi/10.1145/3315508.3329973)
