# 15 — Block Pointer API 完全指南

> **目标**: 深入理解 `tl.make_block_ptr` + `tl.advance` 的完整语义，掌握从手工指针迁移到 block pointer 的最佳实践。
> **前置**: 笔记 01（Triton 编程模型）、笔记 02（内存层级）、至少写过 `phase2_compute/02_matmul_tiled.py`

---

## 0. 为什么需要 Block Pointer？

### 先看老写法（手工指针拼接）

```python
# phase2_compute/02_matmul_tiled.py 的老写法
pid_m = tl.program_id(axis=0)
pid_n = tl.program_id(axis=1)

offs_m = pid_m * BM + tl.arange(0, BM)
offs_n = pid_n * BN + tl.arange(0, BN)
offs_k = tl.arange(0, BK)

acc = tl.zeros([BM, BN], dtype=tl.float32)
for k in range(0, K, BK):
    # 手工拼接地址: 3 行才能描述 "加载 A 的一个 [BM,BK] tile"
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak
    a_mask = (offs_m[:, None] < M) & (k + offs_k[None, :] < K)
    a = tl.load(a_ptrs, mask=a_mask, other=0.0)

    # B 同理: 又 3 行
    b_ptrs = b_ptr + (k + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn
    b_mask = (k + offs_k[:, None] < K) & (offs_n[None, :] < N)
    b = tl.load(b_ptrs, mask=b_mask, other=0.0)

    acc += tl.dot(a, b)
```

**问题**:
- 每个 tile load 要写 3 行（offsets → ptrs → mask → load）
- `[:, None]` / `[None, :]` 的 broadcasting 是常见 bug 来源
- 编译器看到的是"一个 base pointer + 一堆算术表达式"，**无法推理访问模式**
- K 循环中的 `k + offs_k` 会被编译器当作"运行时未知偏移"

### 新写法（Block Pointer）

```python
# phase3_production/01_matmul_block_ptr.py 的新写法
p_a = tl.make_block_ptr(
    base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
    offsets=(pid_m * BM, 0), block_shape=(BM, BK), order=(1, 0))
p_b = tl.make_block_ptr(
    base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
    offsets=(0, pid_n * BN), block_shape=(BK, BN), order=(1, 0))

for k in range(0, K, BK):
    a = tl.load(p_a, boundary_check=(0, 1))  # 一行搞定！
    b = tl.load(p_b, boundary_check=(0, 1))
    acc += tl.dot(a, b)
    p_a = tl.advance(p_a, (0, BK))
    p_b = tl.advance(p_b, (BK, 0))
```

**改进**:
- **代码量减半**: 每个 tile load 从 3 行 → 1 行
- **语义明确**: "从 A 中取一个 BM×BK 的 tile，起始于 (pid_m*BM, 0)"
- **编译器友好**: shape/strides/block_shape 都是编译时已知 → 编译器可以推理访问模式 → 更好的 coalescing、prefetch、TMA 映射

---

## 1. `tl.make_block_ptr` 完整签名

```python
tl.make_block_ptr(
    base,          # 指针: 数据的起始地址（可以是 tl.tensor of pointers）
    shape,         # tuple: 源数据的逻辑形状，如 (M, K)
    strides,       # tuple: 每个维度的 stride（以元素为单位）
    offsets,       # tuple: 当前要访问的 tile 在源数据中的起始位置
    block_shape,   # tuple: 要访问的 tile 大小 (BM, BK)
    order,         # tuple: 线程映射顺序（决定 coalescing）
)
```

### 1.1 每个参数的含义

```
假设我们有一个 4×8 的矩阵 A，想取 (1,2) 位置的 2×3 tile:

    A (4×8):                       取 tile = A[1:3, 2:5]:
    ┌─────────────────────┐         ┌─────────────────────┐
    │ 0  1  2  3  4  5  6 7│         │ ·  ·  ·  ·  ·  ·  · ·│
    │ 8  9 10 11 12 13 14 15│  →    │ ·  · (10 11 12) · · ·│  ← offsets=(1,2)
    │16 17 18 19 20 21 22 23│         │ ·  · (18 19 20) · · ·│  ← block_shape=(2,3)
    │24 25 26 27 28 29 30 31│         └─────────────────────┘
    └─────────────────────┘

    base = A 的起始地址
    shape = (4, 8)           # 源矩阵形状
    strides = (8, 1)         # stride[0]=8 (每行 8 个元素), stride[1]=1
    offsets = (1, 2)         # 从第 1 行第 2 列开始
    block_shape = (2, 3)     # 取 2 行 3 列
```

### 1.2 `order` 参数——最容易搞错的一个

`order` 决定了**线程如何在 block_shape 的各个维度上分布**，直接影响 coalescing。

order=(1, 0): dim 1 是 "innermost"（最内层）
  → 相邻 thread 映射到 dim 1 的相邻位置
  → 如果 dim 1 的 stride=1（列连续），则访问是 coalesced
  → 这是 PyTorch row-major 内存布局的正确选择

order=(0, 1): dim 0 是 innermost
  → 相邻 thread 映射到 dim 0 的相邻位置
  → 如果 dim 0 的 stride=1（行连续），则访问是 coalesced
  → 适合 column-major（Fortran 风格）布局


**直觉理解**：

$A[i, j]$ 的线性地址 $= i \times \text{stride\_am} + j \times \text{stride\_ak}$

如果 $\text{stride\_ak}=1$（列连续）:
  $\text{地址}(A[i, j]) - \text{地址}(A[i, j-1]) = 1$  (列方向连续)
  应该让相邻线程沿 $j$ 方向 → $\text{order}=(1,0)$

如果 $\text{stride\_am}=1$（行连续）:
  $\text{地址}(A[i, j]) - \text{地址}(A[i-1, j]) = 1$  (行方向连续)
  应该让相邻线程沿 $i$ 方向 → $\text{order}=(0,1)$



**选择 `order` 的法则**：

```python
# stride=1 的维度的 index 应该放在 order 的最前面
# PyTorch 默认 row-major: 最后一维 stride=1
# → order 应该以 (ndim-1, ndim-2, ..., 0) 的顺序

# 例: (M, K) 矩阵, row-major, stride_ak=1
order = (1, 0)   # dim 1 (K) stride=1 → order[0]=1

# 例: (B, H, N, D) 4D tensor, row-major, stride_d=1
order = (3, 2, 1, 0)   # dim 3 (D) stride=1 → order[0]=3
```

> 🔧 **Compiler Perspective**: `order` 被翻译为 Triton 的 `BlockedEncoding` 中的 `order` 字段。编译器在生成 PTX 时，根据 order 为每个线程计算起始偏移和步长（`ld.global.v4.f32` 的地址），确保同一个 warp 内的线程访问连续的 HBM 地址 → 一次 128-byte transaction 服务 32 个线程。

---

## 2. `tl.advance` — 零开销的指针移动

```python
p_a = tl.make_block_ptr(base=a_ptr, shape=(M,K), strides=(sa_m,sa_k),
                        offsets=(pid_m*BM, 0), block_shape=(BM,BK), order=(1,0))

# 沿 K 维推进 BK 步
p_a = tl.advance(p_a, (0, BK))
# → offsets 从 (pid_m*BM, 0) 变成 (pid_m*BM, BK)
# → 现在 p_a 指向 A[pid_m*BM:, BK:] 的 BM×BK tile
```

### 2.1 advance 什么都不算

```python
# advance 是纯粹的编译时元数据更新——不生成任何运行时指令
# 等价于你手工写:
offs_k = (k + BK) + tl.arange(0, BK)   # 手工更新 K 偏移
# 但 advance 让编译器知道"这是上一个 tile 的下一个相邻 tile"
# → 编译器可以做 prefetch: 在处理 tile k 时提前加载 tile k+1
```

### 2.2 advance 的维度对应

```python
# make_block_ptr 有几个维度，advance 就要给几个偏移
p = tl.make_block_ptr(..., shape=(M, N, K), block_shape=(BM, BN, BK), order=(2,1,0))
p = tl.advance(p, (dm, dn, dk))  # 三个维度各自推进

# 2D 例子:
p = tl.make_block_ptr(..., shape=(M, K), ...)
p = tl.advance(p, (0, BK))   # M 维不动, K 维推进 BK

p = tl.make_block_ptr(..., shape=(K, N), ...)
p = tl.advance(p, (BK, 0))   # K 维推进 BK, N 维不动
```

### 2.3 重置 block pointer

```python
# 如果你需要在 Pass 2 中重新从头开始扫描:
# 方法: 重新创建 block pointer（Triton 不支持 "rewind"）
p_x_pass1 = tl.make_block_ptr(..., offsets=(row, 0), ...)  # Pass 1
for _ in range(num_tiles):
    load p_x_pass1; advance

p_x_pass2 = tl.make_block_ptr(..., offsets=(row, 0), ...)  # Pass 2: 重新开始
for _ in range(num_tiles):
    load p_x_pass2; advance
```

---

## 3. `boundary_check` — 自动边界处理

```python
# 手工 mask（老写法）:
mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
a = tl.load(a_ptrs, mask=mask, other=0.0)

# boundary_check（新写法）:
a = tl.load(p_a, boundary_check=(0, 1))
# 编译器自动生成: 对 dim 0 和 dim 1 做边界检查
# dim 0 对应 M 维, dim 1 对应 K 维
```

### 3.1 boundary_check 的维度编号

```python
# 维度编号按 make_block_ptr 的 shape 顺序
p = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(sa, sb),
                      offsets=(0, 0), block_shape=(BM, BK), order=(1, 0))
# shape=(M, K):
#   dim 0 = M 维
#   dim 1 = K 维
# boundary_check=(0,):  只检查 M 维边界
# boundary_check=(0,1): 检查 M 和 K 维边界

# 对于 1D block pointer:
p = tl.make_block_ptr(base=x_ptr, shape=(N,), strides=(1,),
                      offsets=(0,), block_shape=(BN,), order=(0,))
# shape=(N,):
#   dim 0 = 唯一的维度
# boundary_check=(0,): 检查 N 维边界
```

### 3.2 性能考虑

boundary_check=(0,1): 每个 tile access 都做边界检查
  → 编译器插入 predicated load 指令
  → 小幅开销（~2-5%），但对正确性是必要的

boundary_check=(): 不检查任何边界
  → 当你确定 tile 完全在边界内时使用
  → 例: M=4096, BM=128, M 恰好整除 BM
  → 省掉 predicate 指令 → 略快

权衡: 
  - 生产代码推荐始终用 boundary_check（安全第一）
  - 只在 profiler 确认 bottleneck 且你 100% 确定整除时才省略


---

## 4. 1D / 2D / 3D Block Pointer 使用场景

### 4.1 1D Block Pointer — Row-wise 操作

```python
# RMSNorm, LayerNorm, 逐行 reduction
# 每行是一个 program, 沿列方向扫描

p_x = tl.make_block_ptr(
    base=x_ptr + pid * stride_x_m,  # 第 pid 行
    shape=(N,),                      # 1D shape: 只有列维
    strides=(stride_x_n,),           # 列 stride
    offsets=(0,),                    # 从第 0 列开始
    block_shape=(BN,),               # 每次加载 BN 列
    order=(0,),                      # 1D: order=(0,)
)

for _ in range(triron.cdiv(N, BN)):
    x = tl.load(p_x, boundary_check=(0,))
    # ... 处理 ...
    p_x = tl.advance(p_x, (BN,))    # 沿列方向推进
```

### 4.2 2D Block Pointer — GEMM / Attention

```python
# GEMM: A tile [BM, BK], B tile [BK, BN]
p_a = tl.make_block_ptr(
    base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
    offsets=(pid_m * BM, 0), block_shape=(BM, BK), order=(1, 0))

# Attention: Q tile [BQ, D_HEAD], K tile [BK, D_HEAD]
p_q = tl.make_block_ptr(
    base=q_ptr + q_offset, shape=(N_CTX, D_HEAD),
    strides=(stride_q_m, stride_q_d),
    offsets=(pid_q * BQ, 0), block_shape=(BQ, D_HEAD_CONST),
    order=(1, 0))
```

### 4.3 3D Block Pointer — Batch 操作

```python
# Triton 3.x 支持 3D block pointer
# 适用于: batched matmul, multi-head attention 同时 tiling

p_a = tl.make_block_ptr(
    base=a_ptr,
    shape=(B, M, K),              # 3D shape
    strides=(stride_b, stride_m, stride_k),
    offsets=(pid_b, pid_m * BM, 0),
    block_shape=(1, BM, BK),      # batch dim 只取 1
    order=(2, 1, 0),              # K innermost
)
```

---

## 5. 编译器视角: Block Pointer → PTX 的完整路径

> 🔧 本节面向有编译器背景的读者

### 5.1 Lowering 流水线

```
Triton Python (make_block_ptr)
  ↓
TTIR: tt.make_tensor_ptr %base[%shape, %strides, %offsets]
  │      {order = array<i32: 1, 0>}
  │      → tt.load %ptr {boundaryCheck = array<i32: 0, 1>}
  ↓
TTGIR: 插入 SharedEncoding / BlockedEncoding 注解
  │      编译器分析 order 和 strides → 确定 coalescing pattern
  │      生成 shared memory staging（如果有 num_stages > 1）
  ↓
LLVM Dialect: 展开为地址计算 + ld.global 指令
  │      根据 BlockedEncoding 计算每个线程的线性偏移
  │      插入 cp.async（如果 num_stages > 1）
  ↓
PTX: ld.global.nc.v4.f32 / cp.async.ca.shared.global
  │      在 SM90+ (Hopper) 上，block_ptr 可映射为 cp.async.bulk (TMA)
```

### 5.2 关键优化点

**优化 1: Compile-time known strides → 更好的地址计算**

```llvm
// 手工指针拼接 → 编译器看到:
%offset = add %base, %computed_offset   ; %computed_offset 运行时未知
%addr = getelementptr %offset, %thread_idx

// block_ptr → 编译器看到:
%tile_base = tt.make_tensor_ptr %base, shape=(M,K), strides=(stride_am,1)
// strides[1]=1 → 编译器知道 dim 1 连续 → 生成 coalesced load
// offsets 的 K 部分在循环中被 advance → 编译器知道每次推进 BK
// → 编译器可以在编译时计算每次 advance 后的地址
```

**优化 2: 已知 block_shape → 更好的 prefetch 调度**

```python
# block_shape=(BM, BK) 在编译时已知
# 编译器可以:
# 1. 计算每个 tile 的地址寄存器需求
# 2. 决定是否在 shared memory 中 staging
# 3. 插入 cp.async 并调度 prefetch 距离
```

### 5.3 TMA 映射的前提条件

```python
# 要让 block_ptr 在 Hopper 上映射为 TMA (cp.async.bulk):
# 条件 1: block_shape 的每个维度必须是 16 bytes 的倍数
#   → fp16: 维度必须是 8 的倍数
#   → fp32: 维度必须是 4 的倍数
# 条件 2: order 必须按 stride 递增排序（TMA 对访问模式有要求）
# 条件 3: boundary_check 的维度也由 TMA 硬件处理

# Triton 3.x 中，即使满足条件，TMA 映射也不是保证的
# 取决于编译器是否决定使用 TMA（受 TRITON_ENABLE_TMA 环境变量影响）
```

---

## 6. 迁移指南: 老写法 → Block Pointer

### 6.1 GEMM: A tile

```python
# 老写法:
offs_m = pid_m * BM + tl.arange(0, BM)
offs_k = k + tl.arange(0, BK)
a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
a = tl.load(a_ptrs, mask=a_mask, other=0.0)

# 新写法:
p_a = tl.make_block_ptr(
    base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
    offsets=(pid_m * BM, 0), block_shape=(BM, BK), order=(1, 0))
a = tl.load(p_a, boundary_check=(0, 1))
p_a = tl.advance(p_a, (0, BK))
```

### 6.2 GEMM: B tile（转置场景）

```python
# 老写法:
offs_k = k + tl.arange(0, BK)
offs_n = pid_n * BN + tl.arange(0, BN)
b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
b = tl.load(b_ptrs, mask=b_mask, other=0.0)

# 新写法:
p_b = tl.make_block_ptr(
    base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
    offsets=(0, pid_n * BN), block_shape=(BK, BN), order=(1, 0))
b = tl.load(p_b, boundary_check=(0, 1))
p_b = tl.advance(p_b, (BK, 0))
```

### 6.3 Store 结果

```python
# 老写法:
c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
tl.store(c_ptrs, acc, mask=c_mask)

# 新写法:
p_c = tl.make_block_ptr(
    base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
    offsets=(pid_m * BM, pid_n * BN), block_shape=(BM, BN), order=(1, 0))
tl.store(p_c, acc, boundary_check=(0, 1))
```

---

## 7. 常见陷阱与调试

### 7.1 order 配错 → 非合并访问

```python
# ❌ 错误: row-major 内存 + order=(0,1) → 线程沿行方向分布 → 非合并
p = tl.make_block_ptr(base=a_ptr, shape=(M, N), strides=(N, 1),
                      offsets=(0, 0), block_shape=(BM, BN), order=(0, 1))
# stride[1]=1 → 列连续 → 应该 order=(1,0)

# ✅ 正确
p = tl.make_block_ptr(base=a_ptr, shape=(M, N), strides=(N, 1),
                      offsets=(0, 0), block_shape=(BM, BN), order=(1, 0))
```

### 7.2 strides 单位是元素，不是 bytes

```python
# ❌ 错误: 把 PyTorch stride（bytes）直接当元素 stride
# PyTorch: a.stride(0) 是 bytes 数
# Triton: strides 是元素数

# ✅ 正确: Triton 接受 PyTorch stride（元素数），因为 Triton 内部处理
p = tl.make_block_ptr(
    base=a_ptr, shape=(M, K),
    strides=(a.stride(0), a.stride(1)),  # PyTorch stride: 元素数
    ...)
```

### 7.3 advance 偏移量不对 → 越界访问

```python
# ❌ 错误: advance 的偏移维度顺序和 block_shape 不一致
p = tl.make_block_ptr(..., block_shape=(BM, BK), order=(1, 0))
p = tl.advance(p, (BK, 0))  # 对 dim 0 推进 BK, dim1 推进 0？不对

# ✅ 正确: advance 偏移按 shape 维度顺序
p = tl.make_block_ptr(base=a_ptr, shape=(M, K), ...)
p = tl.advance(p, (0, BK))   # M 维 +0, K 维 +BK ✓
```

### 7.4 忘记 reset → Pass 2 从错误位置开始

```python
# ❌ 错误: 直接复用 advance 过的 pointer 做 Pass 2
for _ in range(num_tiles):
    x = tl.load(p_x, boundary_check=(0,))  # ok: Pass 1
    p_x = tl.advance(p_x, (BN,))
# p_x 现在指向 N 的末尾！
for _ in range(num_tiles):
    x = tl.load(p_x, boundary_check=(0,))  # ❌ 从错误位置加载！

# ✅ 正确: 重新创建 block pointer
p_x = tl.make_block_ptr(..., offsets=(0,), ...)  # reset
for _ in range(num_tiles):
    x = tl.load(p_x, boundary_check=(0,))  # Pass 2
```

---

## 8. 总结

| 特性 | 手工指针拼接 | Block Pointer |
|------|------------|---------------|
| 代码量 | 每 tile 3-4 行 | 每 tile 1 行 |
| 边界处理 | 手工 mask + other=0.0 | `boundary_check` 自动处理 |
| 地址计算 | 运行时 `add` + `mul` | 编译器静态展开 |
| 编译器优化 | 无法推理访问模式 | 已知 shape/stride/order → 更好调度 |
| TMA 映射 | 不支持 | 满足条件可映射为 `cp.async.bulk` |
| 调试 | mask 容易写错 | boundary_check 维度可能配错 |
| 灵活性 | 可以做任意地址计算 | 只支持规则的矩形 tile |

**使用时机**：
- **始终用 block pointer** 写新代码——它是 Triton 2.1+ 的推荐 API
- 只有在访问模式极不规整（如 gather/scatter）时才回落手工指针
- 如果看到自己写 `[:, None]` / `[None, :]` 和 `mask=..., other=0.0`，就该换成 block pointer

---

## 参考资料

- [Triton Language Reference — Block Pointer](https://triton-lang.org/main/python-api/triton.language.html#triton.language.make_block_ptr)
- `phase3_production/01_matmul_block_ptr.py` — 2D GEMM 示例
- `phase3_production/03_flash_attention_v3.py` — 2D Attention 示例
- `phase3_production/04_fused_rms_norm_residual.py` — 1D Row-wise 示例
- `phase3_production/05_varlen_attention.py` — 2D + 动态边界示例
- [NVIDIA Hopper TMA 文档](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#tensor-memory-access)
