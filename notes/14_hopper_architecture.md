# 14 — Hopper (H100) 架构：新一代 GPU 的新能力

> H100 相比 A100 不是简单的"更快"，而是引入了多项**新的硬件能力**。理解这些是写出 Hopper-optimized kernel 的前提。

---

## 1. H100 vs A100 — 快速对比

| 特性 | A100 (Ampere) | H100 (Hopper) | 变化 |
|------|--------------|---------------|------|
| SM 数量 | 108 | 132 | +22% |
| Peak FP16 TFLOPS | 312 | 989 | **3.2×** |
| Peak FP8 TFLOPS | — | 1979 | **全新** |
| HBM 带宽 | 2.0 TB/s | 3.35 TB/s | +68% |
| HBM 容量 | 80 GB | 80 GB | — |
| L2 Cache | 40 MB | 50 MB | +25% |
| Shared Mem/SM | 164 KB | 228 KB | +39% |
| 最大 Clock | 1.41 GHz | 1.98 GHz | +40% |

**关键观察**: FP16 peak 暴增 3.2× 不仅是多了几个 SM，更因为 Hopper 引入了新的 Tensor Core 设计。

---

## 2. 新能力 #1: wgmma（Warp Group MMA）

### 2.1 问题：老 MMA 的限制

A100 的 mma.sync:
  一个 warp (32 threads) 协作做 M=16, N=8, K=16 的 MMA
  更大 tile 需要分解为多个 MMA 调用

H100 的 wgmma:
  一个 warp group (4 warps = 128 threads) 协作做更大的 MMA
  支持的 size: M=64, N∈{8,16,24,...,256}, K∈{8,16,24,...,256}
  
  好处:
  - 更大的 tile → 更好的数据复用
  - 更少的同步开销（4 warps 一起同步，而非多次单独同步）
  - 异步执行（wgmma 和普通指令可以重叠）

### 2.2 可视化

```
A100 mma.sync (一个 warp 32 threads):
  ┌─── 16 ───┐
  │          │ 8  ← 一个 N 维
  │  MMA op  │
  └──────────┘

H100 wgmma (一个 warp group 128 threads):
  ┌────── 64 ──────┐
  │                │
  │                │ N (可变)
  │    MMA op      │
  │                │
  └────────────────┘
  
  更大的 tile → 每个元素在寄存器中的停留时间更长 → 更多的数据复用
```

### 2.3 在 Triton 中的状态

Triton 3.x: 部分支持 wgmma
  - tl.dot 在某些 layout 下会生成 wgmma
  - 但不暴露 wgmma 的全部控制参数
  - 这是为什么 H100 上 Triton 的 GEMM 还没达到 cuBLAS 级别的原因之一

---

## 3. 新能力 #2: TMA（Tensor Memory Accelerator）

### 3.1 TMA 是什么？

旧方案 (cp.async, A100):
  每个 warp 发出 cp.async 指令 — 异步 global→shared 拷贝
  问题:
  - 每个 warp 独立做地址计算
  - 2D copy 需要多次 1D copy 拼接
  - 边界处理需要自己写 mask

TMA (H100):
  专门的硬件拷贝单元
  能力:
  - 一键 2D tile copy: "把 HBM 中的 [256×64] 矩形区域拷到 shared memory"
  - 自动做地址计算（包括 stride、padding）
  - 自动做边界处理（越界元素自动设 0）
  - 完全不占用计算资源（独立的硬件单元）
  - 支持多维: 1D, 2D, 3D, 4D, 5D tile copy

### 3.2 TMA 的工作原理

传统 copy:
  for each thread:
    compute address = base + i*stride0 + j*stride1
    check bounds
    load data → shared memory
  → 32 threads × address calc + bounds check

TMA copy:
  // 一次调用，硬件完成一切
  tma_copy(
    dst_shared,    // shared memory 目标地址
    src_global,    // HBM 源地址
    tile_shape,    // [256, 64] — 要拷贝的区域
    global_extent, // [4096, 4096] — 全局 tensor 的边界
  )
  // 1 个硬件请求 → 硬件自动完成地址计算、边界处理、数据搬运

### 3.3 在 Triton 中的状态

Triton 3.x: 实验性支持
  - 需要手动设置 TRITON_ENABLE_TMA=1
  - 不是所有 tl.load 都能生成 TMA
  
  Triton 的未来版本会更好地集成 TMA，但目前仍然是实验性的。

---

## 4. 新能力 #3: Thread Block Cluster

### 4.1 概念

A100:
  Thread Block 是最大的协作单元
  Block 之间只能用 global memory 通信（慢）

H100:
  引入 Thread Block Cluster — 多个 block 组成的协作组
  Cluster 内的 block 可以:
  - 共享 shared memory（distributed shared memory）
  - 通过 sync 同步（cluster-wide barrier）
  - 直接访问其他 block 的 shared memory
  
  好处: 可以处理更大的 tile，而无需全部回到 global memory

### 4.2 Distributed Shared Memory

传统:
  Block 0: shared memory [0...228KB]
  Block 1: shared memory [0...228KB]  ← 两个独立的空间

Cluster (H100):
  Cluster 0 (包含 Block 0-3):
    Block 0 可以直接读 Block 1 的 shared memory（通过硬件 interconnect）
    → 相当于一个虚拟的 4×228KB = 912KB 的 shared memory
    → 但访问其他 block 的 shared memory 仍然有额外延迟

### 4.3 在 Triton 里

Triton 目前还没有 Thread Block Cluster 的抽象。
这是未来可能加入的特性。

---

## 5. 新能力 #4: FP8 支持

### 5.1 两种 FP8 格式

$$
\begin{aligned}
\text{E4M3 (更精确):} &\quad 1\text{ sign} + 4\text{ exponent} + 3\text{ mantissa} = 8\text{ bits} \\
&\quad \text{范围: } \pm 448,\ \text{最小正数: } 2^{-6} \approx 0.016 \\
&\quad \text{用于: 前向计算} \\[4pt]
\text{E5M2 (更大范围):} &\quad 1\text{ sign} + 5\text{ exponent} + 2\text{ mantissa} = 8\text{ bits} \\
&\quad \text{范围: } \pm 57344,\ \text{最小正数: } 2^{-14} \approx 0.00006 \\
&\quad \text{用于: 反向传播（梯度范围大但精度要求低）}
\end{aligned}
$$

### 5.2 Block-wise Scaling

$$
\begin{aligned}
&\text{原始数据（fp16）} \rightarrow \text{量化（per-block scaling）} \rightarrow \text{fp8} \\
&\text{GEMM in fp8} \rightarrow \text{输出} \times \text{scale}_A \times \text{scale}_B \rightarrow \text{恢复精度} \\[4pt]
&\text{H100 的 FP8 MMA 自带 scaling:} \\
&D = (A_{\text{fp8}} \times \text{scale}_A) \mathbin{\text{@}} (B_{\text{fp8}} \times \text{scale}_B) + C_{\text{fp32}}
\end{aligned}
$$

---

## 6. H100 上 Triton 的"做不到"清单

下表解释了为什么 H100 上 Triton 还没有达到 cuBLAS 性能:

| 能力 | A100 | H100 | Triton 3.x 支持 |
|------|------|------|----------------|
| mma.sync | ✅ | ✅ | ✅ 完全支持 |
| wgmma | — | ✅ | ⚠️ 部分支持 |
| TMA | — | ✅ | ⚠️ 实验性 |
| Thread Block Cluster | — | ✅ | ❌ 不支持 |
| FP8 MMA | — | ✅ | ⚠️ 部分支持 |
| Warp Specialization | — | ✅ | ❌ 不支持（需 CUTILE） |

结论: 在 A100 上，Triton GEMM 可以接近 cuBLAS 的 85-90%。
在 H100 上，由于无法使用 wgmma/TMA/warp specialization，
Triton 只能达到 cuBLAS 的 65-75%。

---

## 7. 对其他 kernel 类型的影响

### Flash Attention

H100 上的 Flash Attention:
  - wgmma 可以加速 $QK^T$ 和 PV 的 GEMM
  - TMA 可以加速 Q/K/V tile 的加载
  - 但 Triton 无法直接使用这些...

  这就是为什么 flash-attn 3 (Tri Dao, 2024) 用纯 CUDA 写的
  H100 版本比 Triton 版本快 ~2×。

### GEMM

H100 上的 GEMM:
  FP8 + wgmma + TMA + warp specialization
  理论上可以接近 80-85% of peak 989 TFLOPS

  Triton 能达到的: ~65-75% peak
  cuBLAS 能达到的: ~80-85% peak

---

## 8. 参考资料

- [NVIDIA H100 Architecture Whitepaper](https://resources.nvidia.com/en-us-tensor-core)
- [NVIDIA H100 Tensor Core GPU Architecture](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [CUDA Programming Guide — Hopper Features](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#hopper-specific-features)
