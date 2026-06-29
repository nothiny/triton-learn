# 17 — Persistent Kernel 与 Stream-K

> **目标**: 理解两种生产级 work dispatch 模式——Persistent Kernel（固定 grid + atomic 调度）和 Stream-K（K 维动态分解），掌握何时使用每种模式及其 tradeoff。
> **前置**: 笔记 00（GPU 执行模型）、笔记 01（Triton 编程模型）、`phase2_compute/02_matmul_tiled.py`

---

## 0. 问题: 标准 Grid Launch 有什么不足？

### 0.1 Kernel Launch 不是免费的

```python
# 标准做法: grid = 问题规模 / tile 大小
grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
matmul_kernel[grid](a, b, c, ...)

# 每次 launch 都有开销:
#   - CPU → GPU: kernel 参数拷贝 (~1-5 μs)
#   - GPU: 调度、初始化 CTA (~2-10 μs)
#   - 总 launch overhead: ~5-15 μs
```

### 0.2 Grid Utilization 问题

大矩阵 (M=N=4096, BM=BN=128):
  num_ctas = (4096/128) × (4096/128) = 32 × 32 = 1024
  H100 有 132 SM → 每个 SM 平均 ~7.8 个 CTA → 利用率好

小矩阵 (M=N=256, BM=BN=128):
  num_ctas = (256/128) × (256/128) = 2 × 2 = 4
  H100 有 132 SM → 只有 4 个 SM 有工作 → 96% SM 空闲！


### 0.3 静态 K 维划分的负载不均

标准 GEMM: 每个 CTA 做全部 K → 没问题（所有 CTA 工作量相同）
Split-K: 把 K 静态分成 S 份 → 如果 K%S≠0，最后一份少半块
  更严重的问题: 大矩阵 → K=8192, SPLIT_K=8 → 每份 1024
                中矩阵 → K=512,  SPLIT_K=8 → 每份 64
  如果 K 太小: CTA 数远超 K 的工作量 → 浪费


---

## 1. Persistent Kernel: 固定 Grid + Atomic Work Dispatch

### 1.1 核心思想

标准 launch:
  Grid = cdiv(M,BM) × cdiv(N,BN)  → 一个小 CTA 一出生就固定了 (pid_m, pid_n)
  做完一个 tile 就退出

Persistent kernel:
  Grid = num_sms（固定） → 每个 CTA 永驻，循环获取 tile 工作
  做完一个 tile → atomic 获取下一个 → 直到所有 tile 完成

类比:
  标准 launch = 临时工: 来干一个活就走
  Persistent kernel = 全职员工: 常驻在岗位上，不断接新任务


### 1.2 实现

```python
@triton.jit
def persistent_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    work_counter_ptr,    # 全局 atomic counter
    total_mn_tiles,      # = num_m_tiles × num_n_tiles
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Persistent GEMM: 每个 CTA 循环获取 (pid_m, pid_n) 并处理对应 tile。"""
    num_n_tiles = tl.cdiv(N, BLOCK_N)

    # 最坏情况下一个 CTA 做所有 tile → 循环 total_mn_tiles 次
    for _ in range(total_mn_tiles):
        tile_idx = tl.atomic_add(work_counter_ptr, 1)

        if tile_idx < total_mn_tiles:
            pid_m = tile_idx // num_n_tiles
            pid_n = tile_idx % num_n_tiles

            # === 标准的 per-tile GEMM ===
            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

            for k in range(0, K, BLOCK_K):
                offs_k = k + tl.arange(0, BLOCK_K)
                a_ptrs = a_ptr + offs_m[:, None] * stride_am + \
                         offs_k[None, :] * stride_ak
                a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
                a = tl.load(a_ptrs, mask=a_mask, other=0.0)

                b_ptrs = b_ptr + offs_k[:, None] * stride_bk + \
                         offs_n[None, :] * stride_bn
                b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
                b = tl.load(b_ptrs, mask=b_mask, other=0.0)
                acc += tl.dot(a, b)

            c_ptrs = c_ptr + offs_m[:, None] * stride_cm + \
                     offs_n[None, :] * stride_cn
            c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
            tl.store(c_ptrs, acc, mask=c_mask)
        # else: tile_idx >= total_mn_tiles → 没工作了，但循环继续（空转）


def matmul_persistent(a, b, num_ctas=132, BLOCK_M=64, BLOCK_N=128, BLOCK_K=32):
    """约定 num_ctas = SM 数"""
    M, K = a.shape; K2, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    num_m_tiles = triton.cdiv(M, BLOCK_M)
    num_n_tiles = triton.cdiv(N, BLOCK_N)
    total_mn_tiles = num_m_tiles * num_n_tiles
    work_counter = torch.zeros(1, device=a.device, dtype=torch.int32)

    grid = (num_ctas,)  # ← 固定 size，与 M,N 无关

    persistent_gemm_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        work_counter, total_mn_tiles,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    return c
```

### 1.3 关键设计决策

**Grid size = num_sms？**

```python
# 一般设为 SM 数量:
num_ctas = 132  # H100
num_ctas = 108  # A100 (40GB)
num_ctas = 82   # RTX 4090

# 但可以设为 num_sms 的倍数:
num_ctas = num_sms * 2   # 每个 SM 跑 2 个 CTA → 更好的 occupancy
# 权衡: 更多 CTA → 更多 shared memory 需求 → 可能反而降低 occupancy
```

**Triton 不支持 while-break → for 循环 + if 条件**

```python
# 理想中的写法（Triton 不支持）:
while True:
    tile_idx = tl.atomic_add(work_counter_ptr, 1)
    if tile_idx >= total_mn_tiles:
        break      # ← Triton 不支持 break！
    # ... 处理 tile ...

# Triton 中的替代写法:
for _ in range(total_mn_tiles):     # 上限循环
    tile_idx = tl.atomic_add(work_counter_ptr, 1)
    if tile_idx < total_mn_tiles:   # 条件执行
        # ... 处理 tile ...
    # 超出范围 → if 不执行 → 空转（但 CTA 不退出）

# 代价: 所有 CTA 都循环 total_mn_tiles 次
#       即使工作在第 1 轮就分配完了
#       后面的 total_mn_tiles - 1 轮都是空转
# 影响: 对极小矩阵（tiles << total_mn_tiles）浪费严重
```

### 1.4 Persistent Kernel 的性能特征

✅ 优势:
  - 消除 launch overhead（小矩阵场景收益最大）
  - 100% SM utilization（即使 tiles < num_sms）
  - 天然负载均衡（哪个 SM 快就拿下一个 tile）

❌ 劣势:
  - atomic_add 竞争: 所有 CTA 争抢 counter → 小开销（~1 cycle/warp）
  - for 循环空转: 早期完成的 CTA 浪费计算资源
  - 编译限制: 循环体必须容纳所有分支 → 寄存器压力可能更大
  - 不兼容 autotune: grid 是运行时决定的（num_sms），autotune 难以优化

📊 最佳场景:
  - 矩阵较小（tiles < num_sms × 2）
  - 需要 fuse 多个 kernel 到一个 persistent loop 中
  - 序列长度可变的处理（varlen attention）


---

## 2. Stream-K: K 维动态分解

### 2.1 Split-K 的问题

Split-K: 把 K 静态分为 SPLIT_K 份
  Grid = (cdiv(M,BM), cdiv(N,BN), SPLIT_K)
  每个 CTA 固定分配到特定的 K 范围

  问题 1: K 维负载不均
    K=1000, BK=32, SPLIT_K=4:
      CTA_0: K ∈ [0, 256)         ← 8 个 tile
      CTA_1: K ∈ [256, 512)       ← 8 个 tile
      CTA_2: K ∈ [512, 768)       ← 8 个 tile
      CTA_3: K ∈ [768, 1000)      ← 7.25 个 tile
    虽然差距不大，但在极端情况下（小 K × 大 M,N）会有影响

  问题 2: 不适合推理场景（batch=1, M=N=1, K 很大）
    × M=1 → 不需要 split MN
    × K 很大 → 需要 split K
    → 但静态 split K 做了不必要的 partial reduction


### 2.2 Stream-K 的核心思想

Stream-K: K tiles 是 "流" 入的

  不固定 CTA 的 K 范围 → 用 atomic counter 动态分配

  算法:
    1. 每个 CTA 原子地获取一段 K tile 范围（如 4 个 K tiles）
    2. 对这组 K tiles，扫描所有 (M,N) tiles
    3. partial result 通过 atomic_add 累加到全局 C
    4. 回到步骤 1，直到所有 K tiles 处理完

  关键区别:
    Split-K: grid 有 3 维 (pid_m, pid_n, pid_k)
             每个 CTA 处理固定的 (M tile, N tile, K group)

    Stream-K: grid 有 1 维 (num_ctas,)
              每个 CTA 动态获取 K group → 遍历所有 (M,N) tiles


### 2.3 实现

```python
@triton.jit
def stream_k_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    k_counter_ptr,       # K 维 atomic counter
    num_k_tiles,
    num_m_tiles,
    num_n_tiles,
    total_mn_tiles,      # = num_m_tiles × num_n_tiles
    tiles_per_k_group,   # 每个 CTA 一次获取多少个 K tile
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Stream-K GEMM: CTA 获取 K tile 范围 → 处理所有 (M,N) tiles → 重复。
    Grid: (num_ctas,) — 固定的 CTA 池
    """
    for _ in range(num_k_tiles):
        k_tile_start = tl.atomic_add(k_counter_ptr, tiles_per_k_group)

        if k_tile_start >= num_k_tiles:
            pass  # 空转
        else:
            k_tile_end = tl.minimum(k_tile_start + tiles_per_k_group, num_k_tiles)

            # 扫描所有 (M, N) tiles
            for mn_idx in range(total_mn_tiles):
                pid_m = mn_idx // num_n_tiles
                pid_n = mn_idx % num_n_tiles

                offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
                offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
                acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

                # 处理获取到的 K 范围
                for k_tile in range(k_tile_start, k_tile_end):
                    k = k_tile * BLOCK_K
                    offs_k = k + tl.arange(0, BLOCK_K)

                    a_ptrs = a_ptr + offs_m[:, None] * stride_am + \
                             offs_k[None, :] * stride_ak
                    a = tl.load(a_ptrs,
                        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
                        other=0.0)
                    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + \
                             offs_n[None, :] * stride_bn
                    b = tl.load(b_ptrs,
                        mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
                        other=0.0)
                    acc += tl.dot(a, b)

                # Atomic 累加到全局 C
                c_ptrs = c_ptr + offs_m[:, None] * stride_cm + \
                         offs_n[None, :] * stride_cn
                tl.atomic_add(c_ptrs, acc,
                    mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

### 2.4 Stream-K vs Split-K vs 标准 GEMM

```
                        标准 GEMM          Split-K          Stream-K
  ─────────────────────────────────────────────────────────────────
  Grid 维度              2D (M,N)         3D (M,N,K)       1D (num_ctas)
  K 分配                 静态-全量         静态-等分          动态-原子获取
  每 CTA 处理            1 M tile ×       1 M tile ×        K group ×
                         全部 K            1/K 份 K          全部 M×N tiles
  ─────────────────────────────────────────────────────────────────
  atomic 累加            ❌ 不需要         ✅ 需要            ✅ 需要
  C 是否需要清零          ❌                ✅                  ✅
  负载均衡                ✅ 完美           ⚠️ K%split≠0时略差  ✅ 完美
  中小矩阵                ✅ 最好           ❌ 浪费 grid       ✅ 最好
  大 K 矩阵               ✅ 最好           ⚠️ 还行            ⚠️ 还行
  单 token 推理            ❌ grid=(1,1)    ✅ 可用            ✅ 可用
  ─────────────────────────────────────────────────────────────────
  局部性                  ✅ 最好           ✅ 好              ❌ 较差
                                              (每 CTA 固定        (每 CTA 扫描
                                               M,N tile)           全部 M,N tiles)

  局部性差异:
    标准: 外循环 K, 内循环每 CTA 固定 (M,N)
      → A tile 在 shared memory 中跨相邻 K tiles 复用
    Stream-K: 外循环 (M,N), 内循环处理 K group
      → A tile 每切换一个 (M,N) 就要重新加载
      → L2 cache 压力更大
```

---

## 3. 两种模式的融合: Persistent + Stream-K

### 3.1 为什么融合？

Persistent kernel 解决: launch overhead, grid utilization
Stream-K 解决: K 维负载均衡

融合: 
  Grid = num_sms（persistent 的固定 grid）
  每个 CTA 先在 K 维做 stream-K 式的工作获取，再处理 MN tiles
  → 同时获得: 零 launch overhead + K 维负载均衡


### 3.2 实现概要

```python
@triton.jit
def persistent_stream_k_kernel(...):
    # 外循环: persistent work dispatch
    for _ in range(total_mn_tiles):
        tile_idx = tl.atomic_add(work_counter_ptr, 1)
        if tile_idx < total_mn_tiles:
            pid_m = tile_idx // num_n_tiles
            pid_n = tile_idx % num_n_tiles
            offs_m = pid_m * BM + tl.arange(0, BM)
            offs_n = pid_n * BN + tl.arange(0, BN)
            acc = tl.zeros([BM, BN], dtype=tl.float32)

            # 内循环: stream-K 式的 K 动态获取
            k_start = 0
            while k_start < K:
                # 原子获取一段 K
                k_tile = tl.atomic_add(k_counter_ptr, tiles_per_k_group)
                k_actual_start = k_tile * BK
                k_actual_end = min(k_actual_start + tiles_per_k_group * BK, K)

                # 处理这段 K
                for k in range(k_actual_start, k_actual_end, BK):
                    # load A tile, B tile, dot...
                    ...

            # store result (不需要 atomic_add，因为 (M,N) tile 唯一)
            tl.store(c_ptrs, acc, mask=c_mask)
```

---

## 4. Atomic 操作的性能特性

### 4.1 Triton 中的 atomic 操作

```python
# Triton 支持的 atomic 操作:
tl.atomic_add(ptr, val, mask=mask)   # 原子加
tl.atomic_max(ptr, val, mask=mask)   # 原子取最大值
tl.atomic_min(ptr, val, mask=mask)   # 原子取最小值
tl.atomic_cas(ptr, cmp, val)         # Compare-and-Swap
tl.atomic_and(ptr, val, mask=mask)   # 原子与
tl.atomic_or(ptr, val, mask=mask)    # 原子或
tl.atomic_xor(ptr, val, mask=mask)   # 原子异或
tl.atomic_xchg(ptr, val, mask=mask)  # 原子交换
```

### 4.2 性能考虑

atomic_add 的开销:
  - 同一 warp 内无竞争: ~1 cycle（warp-level atomic）
  - SM 内跨 warp 竞争: ~20-30 cycles（L1 cache atomic）
  - 跨 SM 竞争（L2 cache）: ~200+ cycles

Persistent kernel 的 work counter:
  - 132 个 CTA 争抢一个 counter
  - 但在 SM 数量级上竞争不激烈（每个 CTA 做完一个 tile 才抢一次）
  - 实测开销 < 1%

Stream-K 的 partial reduction:
  - 多个 CTA 可能 atomic_add 到同一个 C tile
  - 竞争程度取决于 tiles_per_k_group 和 grid size
  - tiles_per_k_group 越大 → 每个 CTA 的 K 范围越大 → 竞争越少
  - 但越大也意味着负载均衡越粗


---

## 5. 选择指南

```
                    标准 Launch    Persistent    Stream-K     Persistent+StreamK
  ─────────────────────────────────────────────────────────────────────────────
  小矩阵              ⚠️ SM空闲      ✅ 推荐        ⚠️ 过度设计    ✅ 最优
  (tiles < num_sms)
  中矩阵              ✅ 推荐         ✅ 可用        ⚠️ 过度设计    ⚠️ 过度设计
  (tiles ≈ num_sms)
  大矩阵              ✅ 推荐         ✅ 可用        ⚠️ 局部性差    ⚠️ 过度设计
  (tiles >> num_sms)
  单 token 推理       ❌ grid=(1,1)  ⚠️ 可用        ✅ 推荐        ✅ 推荐
  varlen sequences    ❌ 难适配       ✅ 推荐        ⚠️ 可用        ✅ 推荐
  K 维负载不均衡       ⚠️ 还行        ⚠️ 还行        ✅ 推荐        ✅ 推荐
  需要 max 数值精度    ✅ 最好        ⚠️ atomic_add   ⚠️ atomic_add  ⚠️ atomic_add

  经验法则:
    - 大矩阵 + 精度优先 → 标准 launch（无 atomic_add，最精确）
    - 小矩阵 / varlen → persistent kernel
    - 推理场景（K 很大，MN 小）→ Stream-K
    - 生产 pipeline（fuse 多个 kernel）→ persistent kernel
```

---

## 6. 总结

Persistent Kernel:
  核心: 固定 grid + atomic work dispatch
  优点: 消除 launch overhead，100% SM utilization
  限制: Triton 不支持 while-break → for 循环空转

Stream-K:
  核心: K 维动态 work stealing
  优点: K 维完美负载均衡
  限制: 局部性较差（跨 MN 扫描），需要 atomic_add 存结果

共同限制:
  - 不兼容 @triton.autotune（grid 是运行时决定的）
  - 数值精度比标准 launch 略差（atomic_add 的累积误差）
  - Triton 的 while-break 缺失 → 代码更冗长

选择思路:
  1. 先写标准 launch → 看是否满足需求
  2. SM 利用率低或单 token 推理 → 换 persistent
  3. K 维负载不均衡 → 换 Stream-K
  4. 两者都需要 → 融合 persistent + Stream-K


---

## 参考资料

- `phase2_compute/11_matmul_persistent.py` — Persistent GEMM 实现
- `phase2_compute/10_matmul_stream_k.py` — Stream-K GEMM 实现
- [Stream-K Paper (Grelck et al., 2023)](https://arxiv.org/abs/2301.03598)
- [RIPPLE (FlashInfer) Persistent Attention](https://github.com/flashinfer-ai/flashinfer)
