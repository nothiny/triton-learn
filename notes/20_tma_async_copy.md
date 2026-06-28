# 20 — TMA 与异步数据搬运

> **目标**: 理解 GPU 异步数据搬运的演进（synchronous → `cp.async` → TMA），掌握 Triton 的 TMA-ready 编程模式，了解 H100 上极限性能的硬件基础。
> **前置**: 笔记 00（GPU 执行模型）、笔记 02（内存层级）、笔记 19（Block Pointer API）、笔记 10（Hopper 架构）

---

## 0. 从一个问题开始: 为什么 H100 上你的 Triton GEMM 跑不过 cuBLAS？

```
A100 (Ampere) 上:
  Triton tiled GEMM ≈ cuBLAS 的 95-98%
  → "Triton 接近手写 CUDA"

H100 (Hopper) 上:
  Triton tiled GEMM ≈ cuBLAS 的 85-90%
  → "为什么差距变大了？"

答案: TMA（Tensor Memory Accelerator）+ Warp Specialization
  H100 引入了专用的硬件数据搬运单元 TMA
  cuBLAS/CUTLASS 3.x 用 TMA + warp specialization 榨干了 HBM 带宽
  Triton 3.x 对 TMA 的支持仍在实验阶段
```

---

## 1. 异步数据搬运的演进

### 1.1 第一代: 同步 Load（Volta 及之前）

```
传统 ld.global（同步加载）:
  Thread 0: [发起 load] → [等 500 cycles] → [数据到了] → [继续计算]
  所有 32 个线程都等着，计算单元空闲

  时间线:
  ═══Load══╪══════stall (500 cycles)══════╪══Compute══
            ↑ 全体线程啥也不干                ↑
```

**核心问题**: Load latency（~500 cycles）无法被隐藏，因为线程在 load 完成前不能做任何事。

### 1.2 第二代: `cp.async`（Ampere A100 / SM80）

```
cp.async（异步拷贝）:
  硬件提供了一条特殊指令，把数据从 HBM 拷贝到 Shared Memory
  关键特性:
    - 不占用线程寄存器（vs 普通 load 要占用目标寄存器）
    - 线程发起拷贝后可以继续执行其他指令
    - 通过 cp.async.commit_group + cp.async.wait_group 做同步

  时间线:
  ═[cp.async tile 1]══[Compute tile 0]══[cp.async.wait]══[Compute tile 1]══
    ↑ 异步搬运中          ↑ 同时计算！                    ↑ 等 tile 1 就绪

  软件流水线视角:
  num_stages=2 (double buffering):
    Stage 0: 装载 tile k,     计算 tile k-1
    Stage 1: 装载 tile k+1,   计算 tile k
    → 装载和计算重叠 → 隐藏了 ~300 cycles 的 HBM 延迟
```

**Triton 如何使用 `cp.async`**：

```python
# 你只需要设置 num_stages > 1:
@triton.autotune(configs=[
    triton.Config({...}, num_stages=2),  # ← 这行触发 cp.async
    triton.Config({...}, num_stages=3),  # ← 三级流水
], key=['M', 'N', 'K'])
@triton.jit
def matmul_kernel(...):
    # 你不需要写任何 cp.async 代码！
    # Triton 编译器自动:
    #   1. 展开 K 维循环
    #   2. 分析依赖: tile k+1 的 load 不依赖 tile k 的 compute
    #   3. 插入 cp.async 指令替代普通 ld.global
    #   4. 插入 cp.async.commit_group / wait_group 做同步
    #   5. 分配 double/triple buffer 的 shared memory
```

> 🔧 **Compiler Perspective**: 这是经典的 **modulo scheduling**（模调度）问题。编译器展开循环体 → 构建数据依赖图 → 计算 initiation interval（II，两次迭代之间的最小周期）→ 重排指令使得 load 在 compute 之前若干个周期发起。`num_stages` 控制 pipeline depth：`num_stages=2` 意味着两个 tile 的 load 和 compute 可以同时飞行。

### 1.3 第三代: TMA（Hopper H100 / SM90）

```
TMA（Tensor Memory Accelerator）:
  H100 引入的专用硬件数据搬运单元——独立的固定功能硬件

  cp.async:
    每个 warp 自己发起异步拷贝
    仍然占用 warp 的 issue slot（每 cycle 发一条指令）
    数据搬运的地址计算仍由 warp 的 ALU 完成

  TMA:
    独立的硬件单元，在 SM 的 TMA 单元中运行
    完全不占用 warp 的计算资源
    地址计算、边界检查由 TMA 硬件完成
    一次可以搬运一个 2D/3D 矩形 tile（而不是逐行搬运）
    支持 reduction（搬入 SM 的同时做加总）

  类比:
    cp.async = 你在图书馆一边找书一边看书
    TMA      = 有个专门的图书管理员帮你搬书，你只管看
```

---

## 2. TMA 硬件架构

### 2.1 TMA 单元在 SM 中的位置

```
  一个 H100 SM 的内部（简化）:
  ┌─────────────────────────────────────────────┐
  │  Warp Scheduler ×4                           │
  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐       │
  │  │Warp 0│ │Warp 1│ │Warp 2│ │Warp 3│       │
  │  │ 计算  │ │ 计算  │ │ 计算  │ │ 计算  │       │
  │  └──────┘ └──────┘ └──────┘ └──────┘       │
  │                                              │
  │  ┌──────────────────────────────┐            │
  │  │  TMA Unit (硬件数据搬运器)     │            │
  │  │  - 接收 TMA descriptor        │            │
  │  │  - 硬件计算地址和边界          │            │
  │  │  - 发起 cp.async.bulk 事务    │            │
  │  │  - 支持 1D/2D/3D tile copy   │            │
  │  │  - 可选: 搬入时做 reduction    │            │
  │  └──────────────────────────────┘            │
  │                                              │
  │  Shared Memory (228 KB)                      │
  │  ┌──────────────────────────────────────┐    │
  │  │ Buffer 0 │ Buffer 1 │ Buffer 2 │ ...  │    │
  │  └──────────────────────────────────────┘    │
  └─────────────────────────────────────────────┘
             ↕ cp.async.bulk (TMA 事务)
  ┌─────────────────────────────────────────────┐
  │              L2 Cache (50 MB)                 │
  └─────────────────────────────────────────────┘
             ↕
  ┌─────────────────────────────────────────────┐
  │              HBM (80 GB, 3.35 TB/s)          │
  └─────────────────────────────────────────────┘
```

### 2.2 TMA Descriptor — TMA 的"指令"

```
TMA descriptor 是一个内存中的数据结构，描述一次数据搬运:

  struct TmaDescriptor {
    uint64_t source_address;     // HBM 源地址
    uint32_t destination_address; // Shared Memory 目标地址
    uint16_t global_dims[5];     // 全局张量维度
    uint16_t global_strides[4];  // 全局 stride
    uint16_t box_dims[5];        // 要搬运的 tile 大小
    uint16_t elem_stride[4];     // 元素间 stride
    uint32_t elem_size;          // 元素大小 (2=fp16, 4=fp32)
    // ... 更多字段 ...
  };
  // 大小: 128 bytes（恰好一个 cache line）

Triton make_block_ptr → TMA descriptor:
  tl.make_block_ptr(
      base=a_ptr,           → source_address
      shape=(M, K),         → global_dims
      strides=(sa, sk),     → global_strides
      offsets=(pid*BM, 0),  → 起始偏移
      block_shape=(BM, BK), → box_dims
      order=(1, 0))         → elem_stride
```

### 2.3 TMA 支持的 Copy 模式

```python
# 1D copy: 搬一行
# cp.async.bulk [dst], [src], 128  # 搬 128 bytes

# 2D tile copy: 搬一个矩形
# cp.async.bulk.tile.2d [dst], [src], descriptor
# 硬件自动处理: 行间 stride、边界越界、对齐

# 3D tile copy: 搬一个长方体
# cp.async.bulk.tile.3d [dst], [src], descriptor

# TMA store: 从 Shared Memory 写回 HBM
# cp.async.bulk.global.shared [dst], [src]
```

---

## 3. Triton 3.x 的 TMA 支持现状

### 3.1 支持层次

```
Triton 3.6 (当前版本):
  ✅ tl.make_block_ptr — TMA-ready 的数据描述
  ✅ num_stages > 1 → cp.async — 自动软件流水线
  ⚠️ TMA 映射 — 通过 TRITON_ENABLE_TMA=1 环境变量启用
     - 编译器在某些条件下将 block_ptr + load 映射为 cp.async.bulk
     - 不是所有 kernel 都能成功映射
     - 行为是 best-effort，不保证

Triton 未来版本 (3.7+):
  🔮 tl.make_tma_copy — 显式 TMA async copy
  🔮 tl.make_tma_store — 显式 TMA async store
  🔮 tl.create_tma_descriptor — 创建 TMA descriptor
  🔮 可能的 warp specialization 支持
```

### 3.2 TMA 映射的前置条件

```python
# 要让 Triton 编译器将 block_ptr load 映射为 TMA:
# 条件 1: GPU 是 SM90+ (H100, H200, B100/B200)
# 条件 2: block_shape 各维度是 16 bytes 的倍数
#   fp16: BM 和 BK 都要是 8 的倍数
#   fp32: BM 和 BK 都要是 4 的倍数
# 条件 3: order 参数是递增的 stride 顺序
# 条件 4: strides 在编译时已知（是 constexpr）
# 条件 5: 设置了 TRITON_ENABLE_TMA=1

# 满足这些条件的 GEMM kernel:
p_a = tl.make_block_ptr(
    base=a_ptr, shape=(M, K),
    strides=(stride_am, stride_ak),
    offsets=(pid_m * BM, 0),
    block_shape=(BM, BK),   # BM=128(✅), BK=64(✅) → 16 bytes 的倍数
    order=(1, 0))           # stride[1] < stride[0]? ✅
a = tl.load(p_a, boundary_check=(0, 1))
# ↑ 编译器可能把这个 load 映射为 cp.async.bulk.tile.2d
```

### 3.3 验证是否使用了 TMA

```bash
# 方法 1: 检查 PTX 输出
TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 \
python phase3_production/01_matmul_block_ptr.py 2>&1 | \
grep -E "cp.async.bulk|tma"

# 如果看到 cp.async.bulk.tile.2d → TMA 被使用
# 如果只看到 cp.async.ca.shared.global → 普通 cp.async

# 方法 2: 用 ncu 查看 memory workload
ncu --set memory --launch-count 1 \
python phase3_production/01_matmul_block_ptr.py
# 在 "Memory Workload Analysis" 中找 TMA 相关的 metric
```

---

## 4. TMA + Warp Specialization: H100 的终极优化

### 4.1 为什么单靠 TMA 不够？

```
只用 TMA 做数据搬运，warp 同时做计算:
  Producer warp = Consumer warp = 同一个 warp
  → 数据搬运虽然不占 ALU，但仍占 issue slot
  → 寄存器被 load 和 compute 共享，可能因寄存器压力降低 occupancy

TMA + Warp Specialization:
  把 warp 分成两组:
    Producer warps: 专门用 TMA 搬运数据
    Consumer warps: 专门用 wgmma 做计算
  → 完全解耦：搬运和计算同时进行，互相不干扰
  → 每个 warp 的寄存器压力降低 → 更高的 occupancy
```

### 4.2 Warp Specialization 的流水线

```
  Producer Warp 0:   │TMA load│ barrier │TMA load│ barrier │
  Producer Warp 1:   │   TMA load   │ barrier │TMA load│
  ───────────────────┼──────────────────────────────────────
  Consumer Warp 0:   │         │ wgmma │         │ wgmma │
  Consumer Warp 1:   │         │           wgmma          │
  ───────────────────┼──────────────────────────────────────
  Shared Memory:     │Buf0 │Buf1 │Buf2 │Buf0 │Buf1 │Buf2 │
                     └──────────────────────────────────────
                       ↑ 用 shared memory barrier 同步
```

### 4.3 Triton 的限制

```
Triton 的 block-level 编程模型:
  一个 kernel 中所有线程执行相同的代码（SPMD）
  → 无法在 Triton 代码中区分 "producer warp" vs "consumer warp"
  → 编译器理论上可以做 warp-level 调度，但这是一项巨大的工程

  当前状态: Triton 编译器不能做 warp specialization
  这意味着:
    - TMA 可以用（减少地址计算），但生产者-消费者解耦做不到
    - 这是 H100 上 Triton < cuBLAS 的根本原因

  未来可能路径:
    - CUDA CUTLASS 做参考: 手动编写 warp-specialized kernel
    - Triton 可能引入 "warp group" 编程抽象
    - cuTile Python 在 Python 层面支持 warp specialization
```

---

## 5. 实战: 为 TMA 做好准备

### 5.1 写 TMA-friendly 的 Triton 代码

```python
@triton.autotune(
    configs=[
        # TMA-friendly 的 block size 选择:
        # - BM 和 BK 是 16 bytes 的倍数
        # - block_shape 不太大（TMA 一次搬 1-2KB 效率最高）
        triton.Config({"BM": 128, "BN": 128, "BK": 64}, num_warps=8, num_stages=3),
        triton.Config({"BM": 128, "BN": 256, "BK": 64}, num_warps=8, num_stages=3),
        triton.Config({"BM": 256, "BN": 128, "BK": 64}, num_warps=8, num_stages=3),
        triton.Config({"BM": 256, "BN": 256, "BK": 128}, num_warps=8, num_stages=2),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def tma_ready_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # TMA-friendly 的关键:
    # 1. order 正确匹配内存布局
    p_a = tl.make_block_ptr(
        base=a_ptr, shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BM, 0),
        block_shape=(BM, BK), order=(1, 0))  # row-major → order=(1,0) ✅

    p_b = tl.make_block_ptr(
        base=b_ptr, shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BN),
        block_shape=(BK, BN), order=(1, 0))  # row-major → order=(1,0) ✅

    # 2. 使用 boundary_check 让编译器处理边界
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(p_a, boundary_check=(0, 1))
        b = tl.load(p_b, boundary_check=(0, 1))
        acc += tl.dot(a, b)
        p_a = tl.advance(p_a, (0, BK))
        p_b = tl.advance(p_b, (BK, 0))

    p_c = tl.make_block_ptr(
        base=c_ptr, shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BM, pid_n * BN),
        block_shape=(BM, BN), order=(1, 0))
    tl.store(p_c, acc.to(tl.float16), boundary_check=(0, 1))
```

### 5.2 启用 TMA

```bash
# 设置环境变量 + 运行
TRITON_ENABLE_TMA=1 python phase3_production/01_matmul_block_ptr.py

# 对比有无 TMA 的性能差异
python -c "
import os, torch, triton
from triton.testing import do_bench

# 不启用 TMA
os.environ.pop('TRITON_ENABLE_TMA', None)
# ... benchmark ...

# 启用 TMA
os.environ['TRITON_ENABLE_TMA'] = '1'
# ... benchmark ...

# 预期: H100 上 5-15% 带宽提升（主要来自地址计算的节省）
"
```

---

## 6. 性能预期和瓶颈分析

### 6.1 TMA 能提升什么、不能提升什么

```
TMA 能提升的:
  ✅ 地址计算开销: 从 ~10% 寄存器降到 ~0%
  ✅ 边界检查: 硬件直接处理
  ✅ 2D tile copy 效率: 硬件自动处理行间 stride
  ✅ L2 cache bandwidth: TMA 有专用的 L2 cache 端口

TMA 不能提升的（如果瓶颈不在这里）:
  ❌ MMA 计算吞吐: TMA 不影响 Tensor Core 速度
  ❌ Shared memory bandwidth: TMA 不改变 shared memory 速度
  ❌ 寄存器压力: TMA 不减少 compute 部分的寄存器用量
     → 如果 kernel 本身是 compute-bound，TMA 帮助很小

典型场景:
  Large GEMM (M,N ≥ 4096): compute-bound → TMA 提升 3-5%
  Small GEMM (M,N ≤ 512): memory-bound → TMA 提升 10-15%
  Flash Attention: mixed → TMA 提升 5-10%
  Elementwise: memory-bound → TMA 作用有限（没有 tile reuse）
```

### 6.2 H100 上的完整性能栈

```
                        H100 GEMM 性能层级

  ┌─────────────────────────────────────────────────┐
  │  Peak TFLOPS (fp16): ~990 TFLOPS                 │ ← 理论极限
  ├─────────────────────────────────────────────────┤
  │  cuBLAS/CUTLASS 3.x (TMA + warp spec): ~80-90%  │ ← 生产最佳
  ├─────────────────────────────────────────────────┤
  │  Triton + TMA (当前): ~70-80%                    │ ← 我们的目标
  ├─────────────────────────────────────────────────┤
  │  Triton cp.async only: ~65-75%                   │ ← 大多数当前代码
  ├─────────────────────────────────────────────────┤
  │  Triton naive (no pipeline): ~50-60%             │ ← num_stages=1
  ├─────────────────────────────────────────────────┤
  │  PyTorch eager: ~30-50%                          │ ← fused ops only
  └─────────────────────────────────────────────────┘
```

---

## 7. 总结与展望

### 7.1 你现在能做的

1. **用 `tl.make_block_ptr` 替换手工指针拼接** — 这是 TMA 映射的前置条件
2. **选择合适的 `order`** — 确保 coalescing 和 TMA 兼容
3. **block_shape 选 16 bytes 的倍数** — fp16 选 8 的倍数，fp32 选 4 的倍数
4. **设置 `num_stages >= 2`** — 让编译器插入 `cp.async`
5. **关注 Triton 版本更新** — `tl.make_tma_copy` / `tl.make_tma_store` 即将稳定

### 7.2 未来路线

```
当前 (Triton 3.6):
  用 block_ptr + boundary_check + num_stages
  → cp.async (Ampere+) + 可能 TMA (Hopper)

近期 (Triton 3.7+):
  tl.make_tma_copy / tl.make_tma_store
  → 显式 TMA 控制

中期 (Triton 4.0?):
  Warp-level programming API
  → Producer-consumer warp specialization

远期 (Triton + cuTile?):
  cuTile Python 的 warp group 抽象
  → 完全发挥 H100/B100 的硬件能力
```

---

## 参考资料

- [NVIDIA Hopper TMA 文档](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#tensor-memory-access)
- [CUTLASS 3.x TMA + Warp Specialization](https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md)
- [Triton GitHub — TMA tracking issue](https://github.com/triton-lang/triton/issues?q=is%3Aissue+TMA)
- `phase3_production/01_matmul_block_ptr.py` — TMA-ready GEMM 示例
- 笔记 `19_block_pointer_api.md` — Block Pointer 完整指南
- 笔记 `10_hopper_architecture.md` — Hopper 架构总览
