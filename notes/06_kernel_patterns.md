# 06 — 常见 Kernel 模式：Elementwise、Reduce、Scan、Gather、Conv

> 每个模式回答三个问题：**它在 GPU 上到底做了什么**、**Triton 怎么写**、**为什么快（或慢）**。

---

## 1. Elementwise + Fusion — 最简单的模式，最重要的优化

### 1.1 概念

Elementwise: 每个输出元素只依赖**同一位置**的输入元素。

$$
\text{output}[i] = f(\text{input}_1[i], \text{input}_2[i], \ldots)
$$

没有跨元素依赖 → 天然完美并行。GPU 上做 elementwise 唯一的瓶颈是**内存带宽**: 计算密度太低，绝大多数时间花在从 HBM 搬数据上。

### 1.2 Triton 写法

```python
@triton.jit
def gelu_kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    # GELU: x * Φ(x), Φ 用 tanh 近似
    out = 0.5 * x * (1.0 + tl.math.tanh(
        0.7978845608 * (x + 0.044715 * x * x * x)))
    tl.store(out_ptr + offsets, out, mask=mask)
```

就这么简单。`grid = (cdiv(N, BLOCK_SIZE),)` — 每个 program 处理一个 chunk。

### 1.3 关键洞察：Fusion 才是价值所在

单独写一个 elementwise kernel 意义不大——PyTorch 已经够快。**价值在于把多个 op 融合成一个 kernel，消除中间 HBM 往返。**

```
没有 fusion（3 次 kernel launch）:
  HBM → reg → HBM   (gelu)
          HBM → reg → HBM   (dropout)
                  HBM → reg → HBM   (add residual)

有 fusion（1 次 kernel launch）:
  HBM → reg → (gelu → dropout → add) → reg → HBM
                 ↑── 全部在寄存器里完成 ──↑
```

典型融合场景：
- **Activation + Dropout + Residual**: Transformer 的 FFN 里每个 sub-layer 都用
- **LayerNorm + 后续 op**: LN 的统计量（mean, var）已经在寄存器里，直接接着算
- **AdamW 的 weight decay**: `w = w * (1 - lr * weight_decay)` — 纯 elementwise，但必须单独写因为 PyTorch 没有这个 op

性能预期：memory-bound 的 elementwise 融合后，节省的 HBM 带宽直接转化为加速比——**2-3× 是常见的**。

---

## 2. Reduce — 多对一的聚合

### 2.1 概念

Reduce: 沿某个维度把多个元素聚合成一个。

$$
\begin{aligned}
\text{sum}(x_1, \ldots, x_n) &= x_1 + x_2 + \cdots + x_n \\
\text{max}(x_1, \ldots, x_n) &= \max(x_1, x_2, \ldots, x_n)
\end{aligned}
$$

依赖结构是 **fan-in**: 树状收拢，最终只有一个出口。

### 2.2 Triton 怎么写

核心 API: `tl.sum(x, axis=0)` — 对一个 vector 的所有元素求和。

```python
@triton.jit
def row_reduce_kernel(x_ptr, out_ptr, N_ROWS, N_COLS, BLOCK_SIZE: tl.constexpr):
    """
    每行独立做 sum reduce。
    一个 program 处理一行 → 沿列维 (dim=1) reduce
    """
    row = tl.program_id(0)
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # 沿列维分块遍历
    for col_start in range(0, N_COLS, BLOCK_SIZE):
        offsets = row * N_COLS + col_start + tl.arange(0, BLOCK_SIZE)
        mask = (col_start + tl.arange(0, BLOCK_SIZE)) < N_COLS
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        acc += x                              # [BLOCK_SIZE] → 逐元素累加

    result = tl.sum(acc, axis=0)             # BLOCK_SIZE → 标量
    tl.store(out_ptr + row, result)
```

### 2.3 `tl.sum(axis=0)` 在 GPU 上到底做了什么

这是理解 reduce 性能的关键。`tl.sum(v, axis=0)` 把一个 vector 变成一个标量，编译器展开成两级:

**第一级 — Warp 内 shuffle reduce（~5 轮，全在寄存器里）:**

```
stride=16: thread i += thread i+16 的值
stride=8:  thread i += thread i+8 的值
stride=4:  ...
stride=2:  ...
stride=1:  ...
→ 32 个 thread 的值收拢到 thread 0 手里
```

`__shfl_down_sync` 延迟 ~1 cycle/轮，5 轮共 ~5 cycles。

**第二级 — 跨 warp shared memory reduce:**

多个 warp 的 partial 结果写 smem → `__syncthreads` → 一个 warp 对 smem 做 shuffle reduce 得到最终标量。

**所以一个 `tl.sum(axis=0)` 的成本 ≈ 5 cycle (warp 内) + ~20 cycle (smem 跨 warp)。** BLOCK_SIZE 越大，跨 warp 的 smem 开销越大，但 warp 内的 5 轮是固定的。

### 2.4 Reduce 的两种常见结构

**Row-wise reduce** (上面例子): 每行一个标量结果。`grid = (N_ROWS,)`。每个 program 处理一个独立的 "reduce group"。

**Full reduce** (全张量 → 一个标量): `grid = (1,)`，单个 program 遍历整个张量。对大张量 (>128K 元素)，kernel 耗时 >20μs → 不会成为瓶颈。对小张量 (<32K 元素)，kernel launch overhead (~5μs) 不可忽略 → 考虑和前后 op 融合。

**性能定位:** Reduce 是 memory-bound——每个元素只做 ~1 次 FLOP（加法），但需要从 HBM 读一次。优化方向:
1. **减少遍历次数**: 一次遍历同时算 sum 和 max（如 softmax 的前两遍可以合并？不行——max 必须先算完，sum 依赖 max 的稳定化）
2. **融入 producer/consumer**: LayerNorm 里，mean/var 的 reduce 结果直接用于 normalize，不写回 HBM

---

## 3. Scan — 有依赖链的并行化

> 这是最常被问到的模式。分三层讲：Blelloch 算法、Triton 编译器实现、常见错误。

### 3.1 概念和难度

**Scan (prefix sum)**: 每个输出依赖它之前的所有输入。

$$
\text{Input: } [a_1, a_2, a_3, a_4] \quad\longrightarrow\quad
\text{Output: } [a_1,\ a_1 + a_2,\ a_1 + a_2 + a_3,\ a_1 + a_2 + a_3 + a_4]
$$

**Scan 比 Reduce 难在哪:**

| | Reduce | Scan |
|---|---|---|
| 依赖结构 | Fan-in，一个出口 | **Fan-out**，每个输出都要所有前驱 |
| Work complexity | O(n) | O(n log n) — Blelloch 不赢在总 work，赢在延迟 |
| Cross-warp 后 | partial 合并成标量，结束 | partial 合并后**还要写回去** |
| Cross-block | 不需要 | **必须双 kernel**（block sum 走 global memory） |

### 3.2 Blelloch 算法: Up-Sweep + Down-Sweep

数据: `[3, 1, 7, 0, 4, 1, 6, 3]`。Blelloch scan 输出 **exclusive scan**: 位置 `i` 的结果 = 前 `i` 个元素的和。

算法在二叉树视角下最直观——up-sweep 从叶子向根填部分和，down-sweep 从根向叶子分发前缀和:

```
                    ┌──────[0-7]──────┐              ← Level 3 (root)
                    │    total sum    │
           ┌────[0-3]────┐     ┌────[4-7]────┐       ← Level 2
           │  sum(0..3)  │     │  sum(4..7)  │
      ┌─[0-1]──┐   ┌─[2-3]──┐ ┌─[4-5]──┐   ┌─[6-7]──┐  ← Level 1
      │sum(0,1)│   │sum(2,3)│ │sum(4,5)│   │sum(6,7)│
      │  │     │   │  │     │ │  │     │   │  │     │
      3  1     7   0  4     1  6  3                          ← 叶子 = 原始数据
```

<details>
<summary><b>Step 1-4: Up-Sweep（点击展开全部 4 步）</b></summary>

**Step 1 (初始):** `[3, 1, 7, 0, 4, 1, 6, 3]`

**Step 2 (offset=1, 相邻对合并):** `x[1]+=x[0]`, `x[3]+=x[2]`, `x[5]+=x[4]`, `x[7]+=x[6]`
→ `[3, 4, 7, 7, 4, 5, 6, 9]` — Level 1 填满

**Step 3 (offset=2, 跨两对合并):** `x[3]+=x[1]`, `x[7]+=x[5]`
→ `[3, 4, 7, 11, 4, 5, 6, 14]` — Level 2 填满

**Step 4 (offset=4, 跨两组合并):** `x[7]+=x[3]`
→ `[3, 4, 7, 11, 4, 5, 6, 25]` — Root 填满 (`x[7]=25` = 总和)

</details>

<details>
<summary><b>Step 5-8: Down-Sweep（点击展开全部 4 步）</b></summary>

Down-sweep 的核心操作: swap 当前节点和左孩子，然后 propagate。

**Step 5: 清零 Root.** `x[7] = 0` → `[3, 4, 7, 11, 4, 5, 6, 0]`

**Step 6 (offset=4):** `x[3]↔x[7]`, `x[7]+=x[3]` (old) → `[3, 4, 7, 0, 4, 5, 6, 11]`

**Step 7 (offset=2):** Pair (1,3): `x[1]↔x[3]`, `x[3]+=4` → x[3]=4. Pair (5,7): `x[5]↔x[7]`, `x[7]+=5` → x[7]=16
→ `[3, 0, 7, 4, 4, 11, 6, 16]`

**Step 8 (offset=1):** 4 对 swap+propagate →
**`[0, 3, 4, 11, 11, 15, 16, 22]`** ✓

</details>

验证: `exc_scan[5] = 15 = 3+1+7+0+4 = 前5个元素和` ✓

### 3.3 Triton 的正确写法

```python
# 一行搞定 — 编译器自动展开
x = tl.associative_scan(x, 0, lambda a, b: a + b)
```

编译器把它展开成两层:

**Warp 内 (register-level):** `__shfl_up_sync`, stride=1,2,4,8,16，5 轮后每个 warp 得到 32 个元素的 inclusive scan。延迟 ~1 cycle/轮。

**Warp 间 (shared memory):** 各 warp 的 warp_sum 写 smem → `__syncthreads` → thread 0 对 smem 做 scan → 把前置 warp 的累加值加回本 warp。**这是和 reduce 的关键区别: reduce 写 smem 就结束，scan 必须写回去。**

### 3.4 常见错误: `tl.where` 不能做 shift

```python
# ❌ 错误 — 这段代码不是 Blelloch scan
for offset in [1, 2, 4, 8, 16]:
    prev = tl.where(tl.arange(0, BLOCK_SIZE) >= offset, x, tl.zeros_like(x))
    x = x + prev  # thread i 读到的是 x[i]，不是 x[i-offset]！
```

`tl.where(mask, x, 0)` 只在 `x[i]` 和 `0` 之间选——thread `i` 永远读不到 thread `i-offset` 的值。**Triton 没有 `tl.shift` 原语。** 跨线程数据移动只有三条路: warp shuffle（≤32 threads）、`tl.load` 偏移地址（走 smem）、以及 `tl.associative_scan`（编译器帮你选）。

### 3.5 跨 Block Scan

Block 之间没有共享寄存器/smem，必须通过 global memory 传 block sum。至少需要 **两个 kernel**:

```
Kernel 1: 各 block 做 block 内 scan + 输出 block_sum
Kernel 2: 对 block_sums 做 scan（数据量小，单 block 搞定），
          然后把前置 block 的累加值加回各 block 内部
```

单 kernel 内 `for block_start in range(0, N, BLOCK_SIZE)` 做不到这件事——Block 0 的 sum 无法在 kernel 执行期间传给 Block 1。

跨 block scan 的推荐方案: 用 CUB 的 `DeviceScan`，或如果数据量不大（<1024），直接用 PyTorch `torch.cumsum`。

---

## 4. Gather / Scatter — 非连续内存访问

### 4.1 概念

两个互为逆操作的模式，都在做"根据索引数组重新排列数据":

$$
\begin{aligned}
\text{Gather: } &\text{out}[i] = \text{inp}[\text{idx}[i]] \\
\text{Scatter: } &\text{out}[\text{idx}[i]] = \text{inp}[i]
\end{aligned}
$$

### 4.2 Gather 的性能本质

Gather 的访存模式由 `idx` 决定。GPU 的 warp 内 32 个 thread 同时发起 load——如果 `idx` 是连续的，thread 0→addr 0x100, thread 1→addr 0x104，一个 128B transaction 全搞定（coalesced）。如果 `idx` 是随机的，32 个 thread 访问 32 个不同 cache line → 32 次 transaction → **带宽利用率 ~3%**。

这就是为什么 embedding lookup（gather 的主要应用）在推荐系统中是瓶颈——大 embedding table + 随机 batch。

### 4.3 Triton 写法

```python
@triton.jit
def gather_kernel(inp_ptr, idx_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    idx = tl.load(idx_ptr + offs, mask=mask, other=0)
    # 关键: 用 idx 做指针偏移 → 每个 thread 访问不同地址
    val = tl.load(inp_ptr + idx, mask=mask, other=0.0)
    tl.store(out_ptr + offs, val, mask=mask)
```

### 4.4 Scatter 的额外问题: 写冲突

Scatter 里 `out[idx[i]] = inp[i]` ——多个 thread 可能同时写同一个 `out` 位置。如果不冲突（如 permute），用普通 `tl.store`；如果可能冲突（如 histogram），必须用 `tl.atomic_add`，代价很高。

---

## 5. Convolution — 把空间归约映射到 GEMM

### 5.1 三种策略

| 策略 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| Direct conv | 7 层嵌套循环，直接算 | 内存友好 | cache 不友好，难优化 |
| Im2col + GEMM | 把 sliding window 展开成矩阵乘 | 可复用 GEMM 优化 | 内存膨胀 K²× |
| Winograd / FFT | 数学变换减少乘法次数 | 3×3 conv 理论最优 | 精度损失，不通用 |

### 5.2 Triton 中做 Im2col Conv

核心: 每个 program 负责一个输出 tile，在 KH×KW 维上做 reduction。

```python
@triton.jit
def conv2d_kernel(inp_ptr, w_ptr, out_ptr,
                  H, W, C_IN, C_OUT, K,
                  BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    # 计算这个 program 负责的输出位置
    h_out = pid // ((W - K + 1) * C_OUT)
    w_out = (pid // C_OUT) % (W - K + 1)
    c_out = pid % C_OUT
    acc = 0.0

    for kh in range(K):
        for kw in range(K):
            # 输入 feature map 的对应位置
            h_in = h_out + kh
            w_in = w_out + kw
            inp_off = (h_in * W + w_in) * C_IN
            w_off = (kh * K + kw) * C_IN * C_OUT + c_out
            for c_in in range(0, C_IN, BLOCK_SIZE):
                offs = c_in + tl.arange(0, min(BLOCK_SIZE, C_IN - c_in))
                inp = tl.load(inp_ptr + inp_off + offs)
                w = tl.load(w_ptr + w_off + offs * C_OUT)
                acc += tl.sum(inp * w)

    tl.store(out_ptr + pid, acc)
```

性能通常 memory-bound（算术强度 ≈ 0.5-0.9 FLOP/byte）。

---

## 6. 决策树: 看到一个问题，选哪个模式？

```
你的计算长什么样？

├── output[i] = f(input[i])                    → Elementwise
│    没有跨元素的依赖。考虑和相邻 op 融合。
│
├── result = aggregate(input, dim=d)           → Reduce
│    多对一的聚合。用 tl.sum/tl.max(axis=d)。
│    memory-bound，考虑融入 producer/consumer。
│
├── C = A @ B (or A @ B^T)                     → GEMM
│    compute-bound (大尺寸时)。用 tiled matmul +
│    autotune + num_stages。
│
├── output[i] 依赖 output[i-1]                 → Scan
│    block 内用 tl.associative_scan。
│    跨 block 用双 kernel 或 CUB。
│
├── output[i] = input[index[i]]                → Gather
│    根据 index 分布，可能 coalesced 也可能完全随机。
│    随机情况下接受低带宽利用率。
│
└── sliding window over spatial dims           → Convolution
    小 kernel (3×3) 用 Winograd，否则 Im2col+GEMM。
    depthwise conv 是 memory-bound 特殊情况。
```

---

## 参考资料

- [Triton Tutorials — 官方示例](https://triton-lang.org/main/getting-started/tutorials/)
- [GPU Gems 2, Ch39 — Parallel Prefix Sum (Blelloch scan 经典)](https://developer.nvidia.com/gpugems/gpugems2/part-iv-general-purpose-computation-gpus-primer/chapter-39-parallel-prefix-sum-scan-cuda)
- [CUTLASS — Implicit GEMM for Convolution](https://github.com/NVIDIA/cutlass/blob/main/media/docs/implicit_gemm_convolution.md)
