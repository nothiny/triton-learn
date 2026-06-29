# 06 — 常见 Kernel 模式：Reduce、Scan、Gather、Convolution

> 除了 elementwise 和 GEMM，GPU 上还有很多常见的计算模式。这篇整理了在 Triton 中实现这些模式的惯用写法。

---

## 1. Reduce（归约）— 多对一的聚合

### 1.1 概念

Reduce: 把多个元素合并为一个（或少数几个）

常见 reduce:

$$
\begin{aligned}
\text{sum}(x) &: x_1 + x_2 + \cdots + x_n \\
\text{max}(x) &: \max(x_1, x_2, \ldots, x_n) \\
\text{argmax}(x) &: \text{最大值的索引} \\
\text{mean}(x) &: \frac{\text{sum}(x)}{n}
\end{aligned}
$$

Reduce 的本质: "沿某个维度聚合，产出更小的结果"

### 1.2 Triton 实现：Row-wise Softmax

```python
# 每行独立做 softmax — 每行是一个 reduce group
@triton.jit
def reduce_kernel(x_ptr, out_ptr, N_COLS, BLOCK_SIZE: tl.constexpr):
    """
    逐行 softmax: 沿列维（dim=1）做 reduce
    """
    row_idx = tl.program_id(0)  # 每个 program 处理一行
    
    # Step 1: find max（沿列维 reduce，用 max）
    row_max = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
    for block_start in range(0, N_COLS, BLOCK_SIZE):
        offsets = row_idx * N_COLS + block_start + tl.arange(0, BLOCK_SIZE)
        mask = (block_start + tl.arange(0, BLOCK_SIZE)) < N_COLS
        x = tl.load(x_ptr + offsets, mask=mask, other=float('-inf'))
        row_max = tl.maximum(row_max, x)  # per-block max
    global_max = tl.max(row_max, axis=0)  # final reduce across blocks
    
    # Step 2: compute sum（沿列维 reduce，用 sum）
    row_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for block_start in range(0, N_COLS, BLOCK_SIZE):
        offsets = row_idx * N_COLS + block_start + tl.arange(0, BLOCK_SIZE)
        mask = (block_start + tl.arange(0, BLOCK_SIZE)) < N_COLS
        x = tl.load(x_ptr + offsets, mask=mask, other=float('-inf'))
        row_sum += tl.exp(x - global_max)
    global_sum = tl.sum(row_sum, axis=0)
    
    # Step 3: normalize + write
    for block_start in range(0, N_COLS, BLOCK_SIZE):
        offsets = row_idx * N_COLS + block_start + tl.arange(0, BLOCK_SIZE)
        mask = (block_start + tl.arange(0, BLOCK_SIZE)) < N_COLS
        x = tl.load(x_ptr + offsets, mask=mask, other=float('-inf'))
        result = tl.exp(x - global_max) / global_sum
        tl.store(out_ptr + offsets, result, mask=mask)

# [COMPILER] tl.max(x, axis=0) 编译为:
# 1. warp shuffle reduction (fast, in-register)
# 2. shared memory reduction (cross-warp, slower)
```

### 1.3 Reduce 的性能要点

1. Reduce 通常是 memory-bound（很少 FLOPs per element）
   → 优化方向: 减少 HBM 遍历次数

2. tl.sum(axis=0) 跨 block 需要 shared memory
   → BLOCK_SIZE 太大意味着更多 reduce 开销

3. 对于大 reduction，可以考虑:
   - 多级 reduce（block-level → warp-level）
   - 使用 shared memory 缓存中间结果
   - 与后续 op 融合（如 fused softmax）

---

## 2. Scan（扫描）— 从 Blelloch 算法到 Triton 实现

> 分三层讲：**Blelloch 算法本身**（8 步交互演示）、**Triton 编译器怎么实现它**（warp shuffle + shared memory）、**你的代码为什么是错的**。

### 2.1 概念：Scan vs Reduce

**Scan (prefix sum / cumulative sum)** — 每个输出依赖它之前的所有输入:

$$
\begin{aligned}
\text{Input}:&\ [a_1, a_2, a_3, a_4, a_5] \\
\text{Output}:&\ [a_1,\ a_1 + a_2,\ a_1 + a_2 + a_3,\ a_1 + a_2 + a_3 + a_4,\ a_1 + a_2 + a_3 + a_4 + a_5]
\end{aligned}
$$

应用场景: LayerNorm 的 mean/variance、排序/基数排序、Attention 的 causal masking、beam search。

**Scan 比 Reduce 难在哪？**

| | Reduce | Scan |
|---|---|---|
| 依赖结构 | Fan-in（收拢），只有一个出口 | Fan-out（扇出），每个输出都要所有前驱 |
| Work complexity | O(n) | O(n log n) — Blelloch 赢在延迟，不赢在总 work |
| Cross-warp | partial 写到 smem 合并成标量 | partial 写到 smem 后**还要写回去**（每个 thread 都要输出） |
| Cross-block | 不需要（标量结果直接用） | **必须两个 kernel**（block sum 通过 global memory 传递） |

---

### 2.2 第一层：Blelloch 算法 — 8 步交互演示

> 数据: `[3, 1, 7, 0, 4, 1, 6, 3]`（8 个元素，`BLOCK_SIZE=8`）

Blelloch scan 分两个阶段，输出 **exclusive scan**:
- **Up-Sweep（3 步）**: 从叶子向根构建部分和树
- **Down-Sweep（4 步）**: 从根向叶子分发前缀和

下面是二叉树视角。树中每个节点 `[i-j]` 存储它覆盖区间的部分和（up-sweep 时从下往上填，down-sweep 时从上往下修正）。

```
                    ┌──────[0-7]──────┐              ← Level 3 (root)
                    │    total sum    │
           ┌────[0-3]────┐     ┌────[4-7]────┐       ← Level 2
           │  sum(0..3)  │     │  sum(4..7)  │
      ┌─[0-1]──┐   ┌─[2-3]──┐ ┌─[4-5]──┐   ┌─[6-7]──┐  ← Level 1
      │sum(0,1)│   │sum(2,3)│ │sum(4,5)│   │sum(6,7)│
      │  │     │   │  │     │ │  │     │   │  │     │
idx:  0  1     2   3  4     5 6  7     8   9 10    11   (概念树节点编号)
val:  3  1     7   0  4     1 6  3     ←  ←  ←  ←  ←   原始数据 (叶子)
```

---

<details>
<summary><b>Step 1: 初始状态</b> — 叶子节点 = 原始数据</summary>

只有叶子有值。内部节点尚未计算。

```
                    ┌──────[0-7]──────┐
                    │       ?         │
           ┌────[0-3]────┐     ┌────[4-7]────┐
           │     ?       │     │     ?       │
      ┌─[0-1]──┐   ┌─[2-3]──┐ ┌─[4-5]──┐   ┌─[6-7]──┐
      │   ?    │   │   ?    │ │   ?    │   │   ?    │
      │  │     │   │  │     │ │  │     │   │  │     │
      3  1     7   0  4     1  6  3     ←  ←  ←  ←  ←
```

数组: `[3, 1, 7, 0, 4, 1, 6, 3]`

</details>

<details>
<summary><b>Step 2: Up-Sweep offset=1</b> — 相邻两两合并 → 填满 Level 1</summary>

线程 `i` 满足 `(i+1) % 2 == 0` 的，把 `x[i-1]` 加到 `x[i]` 上。

```
      ┌─[0-1]──┐   ┌─[2-3]──┐ ┌─[4-5]──┐   ┌─[6-7]──┐
      │   4    │   │   7    │ │   5    │   │   9    │    ← Level 1 填满
      │ ↗│     │   │ ↗│     │ │ ↗│     │   │ ↗│     │
      3  1     7   0  4     1  6  3
       (1+3)     (0+7)     (1+4)     (3+6)
```

- `x[1] += x[0]` → 1+3=**4**
- `x[3] += x[2]` → 0+7=**7**
- `x[5] += x[4]` → 1+4=**5**
- `x[7] += x[6]` → 3+6=**9**

数组: `[3, 4, 7, 7, 4, 5, 6, 9]`（加粗为本次变化）

</details>

<details>
<summary><b>Step 3: Up-Sweep offset=2</b> — 跨两对合并 → 填满 Level 2</summary>

线程 `i` 满足 `(i+1) % 4 == 0` 的，把 `x[i-2]` 加到 `x[i]` 上。

```
           ┌────[0-3]────┐     ┌────[4-7]────┐
           │     11      │     │     14      │         ← Level 2 填满
      ┌─[0-1]──┐   ┌─[2-3]──┐ ┌─[4-5]──┐   ┌─[6-7]──┐
      │   4    │   │   7 ↗  │ │   5    │   │   9 ↗  │
      3  1     7   0  4     1  6  3
                  (7+4)            (9+5)
```

- `x[3] += x[1]` → 7+4=**11**
- `x[7] += x[5]` → 9+5=**14**

数组: `[3, 4, 7, 11, 4, 5, 6, 14]`

</details>

<details>
<summary><b>Step 4: Up-Sweep offset=4</b> — 跨两组合并 → 填满 Root</summary>

线程 `i` 满足 `(i+1) % 8 == 0` 的，把 `x[i-4]` 加到 `x[i]` 上。

```
                    ┌──────[0-7]──────┐
                    │       25        │                    ← Root 填满
           ┌────[0-3]────┐     ┌────[4-7]────┐
           │     11      │     │     14 ↗    │
      ┌─[0-1]──┐   ┌─[2-3]──┐ ┌─[4-5]──┐   ┌─[6-7]──┐
      │   4    │   │   7    │ │   5    │   │   9    │
      3  1     7   0  4     1  6  3
                                     (14+11)
```

- `x[7] += x[3]` → 14+11=**25**（全部元素总和）

数组: `[3, 4, 7, 11, 4, 5, 6, 25]`

</details>

---

**Up-Sweep 完成。** `x[7] = 25` = 全部元素总和。树中每个内部节点 `[i-j]` 都存在其覆盖区间内**某个位置**: 节点 `[0-3]→x[3]=11`, 节点 `[0-7]→x[7]=25`。

---

<details>
<summary><b>Step 5: 清零 Root</b> — 准备 Down-Sweep</summary>

Down-Sweep 之前先把 `x[7]` 清零。这个 0 将向下传播。

```
                    ┌──────[0-7]──────┐
                    │       0         │                    ← Root 清零
           ┌────[0-3]────┐     ┌────[4-7]────┐
           │     11      │     │     14      │
      ┌─[0-1]──┐   ┌─[2-3]──┐ ┌─[4-5]──┐   ┌─[6-7]──┐
      │   4    │   │   7    │ │   5    │   │   9    │
      3  1     7   0  4     1  6  3
```

- `x[7] = 0`

数组: `[3, 4, 7, 11, 4, 5, 6, 0]`

</details>

<details>
<summary><b>Step 6: Down-Sweep offset=4</b> — Root 向 Level 2 分发</summary>

线程 `i` 满足 `(i+1) % 8 == 0` 的，swap 并 propagate:
- `tmp = x[i-4]`, `x[i-4] = x[i]`, `x[i] += tmp`

```
                    ┌──────[0-7]──────┐
                    │       0         │
           ┌────[0-3]────┐     ┌────[4-7]────┐
           │  0  │       │     │  11  │       │           ← [0-3]清零, [4-7]=11
      ┌─[0-1]──┐   ┌─[2-3]──┐ ┌─[4-5]──┐   ┌─[6-7]──┐
      │   4    │   │   7    │ │   5    │   │   9    │
      3  1     7   0  4     1  6  3
      ↑                   ↑
  x[3]=0              x[7]=11
```

- `tmp = x[3] = 11` → `x[3] = x[7] = 0` → `x[7] = 0 + 11 = 11`

含义: `x[7]` 现在是 `[0-3]` 区间和(11) + 0，即 exc_scan 中 index 7 的前缀和。

数组: `[3, 4, 7, 0, 4, 5, 6, 11]`

</details>

<details>
<summary><b>Step 7: Down-Sweep offset=2</b> — Level 2 向 Level 1 分发</summary>

线程 `i` 满足 `(i+1) % 4 == 0` 的，swap 并 propagate。

```
           ┌────[0-3]────┐     ┌────[4-7]────┐
           │  0  │       │     │  11  │       │
      ┌─[0-1]──┐   ┌─[2-3]──┐ ┌─[4-5]──┐   ┌─[6-7]──┐
      │ 0 │    │   │ 4 │    │ │11 │    │   │16 │    │    ← Level 1 更新
      3  1     7   0  4     1  6  3
      ↑       ↑      ↑       ↑
    x[1]=0  x[3]=4  x[5]=11 x[7]=16
```

- Pair 1: `tmp=x[1]=4` → `x[1]=x[3]=0` → `x[3]=0+4=4`
- Pair 2: `tmp=x[5]=5` → `x[5]=x[7]=11` → `x[7]=11+5=16`

数组: `[3, 0, 7, 4, 4, 11, 6, 16]`

</details>

<details>
<summary><b>Step 8: Down-Sweep offset=1</b> — Level 1 向叶子分发 → 完成！</summary>

线程 `i` 满足 `(i+1) % 2 == 0` 的，swap 并 propagate。

```
      ┌─[0-1]──┐   ┌─[2-3]──┐ ┌─[4-5]──┐   ┌─[6-7]──┐
      │ 0 │ 3  │   │ 4 │11  │ │11 │15  │   │16 │22  │    ← 叶子得到最终值
      0  3     4   11 11 15  16 22                         ← exclusive scan 结果
```

- `x[0]↔x[1]`: `x[0]=0, x[1]=3`
- `x[2]↔x[3]`: `x[2]=4, x[3]=11`
- `x[4]↔x[5]`: `x[4]=11, x[5]=15`
- `x[6]↔x[7]`: `x[6]=16, x[7]=22`

**最终结果（exclusive scan）: `[0, 3, 4, 11, 11, 15, 16, 22]`**

</details>

---

**验证:** 对原始输入 `[3,1,7,0,4,1,6,3]`

| i | exc_scan[i] | 含义 | 验证 |
|---|---|---|---|
| 0 | 0 | 前 0 个元素的和 | 0 ✓ |
| 1 | 3 | 前 1 个元素的和 | 3 ✓ |
| 2 | 4 | 前 2 个元素的和 | 3+1=4 ✓ |
| 3 | 11 | 前 3 个元素的和 | 3+1+7=11 ✓ |
| 4 | 11 | 前 4 个元素的和 | 3+1+7+0=11 ✓ |
| 5 | 15 | 前 5 个元素的和 | 3+1+7+0+4=15 ✓ |
| 6 | 16 | 前 6 个元素的和 | 3+1+7+0+4+1=16 ✓ |
| 7 | 22 | 前 7 个元素的和 | 3+1+7+0+4+1+6=22 ✓ |

---

### 2.3 第二层：Triton 编译器怎么实现 Scan

Triton 提供 `tl.associative_scan`，一行代码搞定:

```python
x = tl.associative_scan(x, 0, lambda a, b: a + b)
```

编译器把它展开成两层:

**Layer 1 — Warp 内 (register-level, 最快):**

使用 `__shfl_up_sync`（PTX warp shuffle），stride = 1, 2, 4, 8, 16，共 **5 轮**:

```
Round 0 (stride=1):  thread i 从 thread i-1 拿值, 相加 → 相邻合并
Round 1 (stride=2):  thread i 从 thread i-2 拿值, 相加 → 跨2合并
Round 2 (stride=4):  thread i 从 thread i-4 拿值, 相加 → 跨4合并
Round 3 (stride=8):  thread i 从 thread i-8 拿值, 相加 → 跨8合并
Round 4 (stride=16): thread i 从 thread i-16 拿值, 相加 → 跨16合并
```

5 轮后每个 warp 得到 32 个元素的 **inclusive scan**。`__shfl_up_sync` 延迟 ~1 cycle（比 shared memory 快 ~20×），这是 warp 级 scan 极快的原因。

**Layer 2 — Warp 间 (shared memory):**

Warps 之间不能 shuffle，必须通过 shared memory 传递数据:

1. 各 warp 最后一个 thread 把 **warp_sum** 写入 smem
2. `__syncthreads` 后，thread 0 对 smem 里的 warp_sums 做一次 warp 级 scan
3. 把 smem 里**前置 warp 的累加值**加回到本 warp 每个元素上

这和 Reduce 的关键区别: **Reduce 写完 smem 就结束了（只需要标量结果）; Scan 写完 smem 后还必须写回去，因为每个 thread 都要自己的输出。**

---

### 2.4 第三层：为什么你手写的代码是错的

原始代码:

```python
for offset in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
    prev = tl.where(tl.arange(0, BLOCK_SIZE) >= offset,
                    x, tl.zeros_like(x))
    x = x + prev  # ← 这行是错的
```

**根本错误: `prev` 并没有做 shift。**

`tl.where(tl.arange(0, BLOCK_SIZE) >= offset, x, tl.zeros_like(x))` 的意思是:
- thread `i` 如果 `i >= offset` → 取 `x[i]`
- thread `i` 如果 `i < offset` → 取 `0`

所以 thread `i` 读到的永远是**自己的** `x[i]`，不是 `x[i - offset]`。最终效果:

```python
x = x + prev  # x[i] + x[i] (if i >= offset else x[i] + 0)
              # = 2*x[i] (if i >= offset) else x[i]
              # 而不是 x[i] + x[i - offset] !
```

这不是 bug 的变种 —— 它做了一件**完全不同的事**。Blelloch scan 要求 thread `i` 能读到 thread `i-offset` 的值，即**跨线程数据移动**。`tl.where` 只做 mask 选择，不具备跨线程能力。

**Triton 没有 `tl.shift` 原语。** 做跨线程数据移动只有三条路:

| 方式 | 适用场景 | 延迟 |
|------|---------|------|
| `__shfl_up_sync` (warp shuffle) | warp 内 (≤32 threads) | ~1 cycle |
| `tl.load` 配合偏移地址 | 任意范围 | ~20 cycles (smem) |
| `tl.associative_scan` | 任意 scan 操作 | 编译器自动选上面两种 |

**正确做法:**

```python
# 一行搞定 — 编译器自动展开 warp shuffle + smem
x = tl.associative_scan(x, 0, lambda a, b: a + b)
```

---

### 2.5 跨 Block Scan 的挑战

上面讲的都是 **block 内** scan（`N ≤ BLOCK_SIZE`）。跨 block 的 scan（`N > BLOCK_SIZE`）更麻烦:

相邻 block 之间无法共享寄存器或 smem，必须通过 **global memory** 传递 block 前缀和。通常需要**两个 kernel**:

```
Kernel 1: 各 block 独立做 block 内 scan，同时输出 block_sum
          block 0 → scan([a₁..aₙ]) = [p₁..pₙ], sum = S₀
          block 1 → scan([b₁..bₙ]) = [q₁..qₙ], sum = S₁
          ...

Kernel 2: 对 block_sums [S₀, S₁, ...] 做 scan → [P₀, P₁, ...]
          然后修正各 block 内的值:
          block 1 的每个元素 += P₀
          block 2 的每个元素 += P₁
          ...
```

这是你代码里 `for block_start in range(0, N_COLS, BLOCK_SIZE)` 这种单 kernel 循环做不到的事——它只能做 block 内 scan，跨 block 的前缀依赖链无法在一个 kernel 内解决。

**替代方案:** 如果 scan 长度不大（<1024），直接用 PyTorch `torch.cumsum`；如果必须 GPU 上做大 scan，用 CUB 的 `DeviceScan`（Triton 目前无内置跨 block scan 支持）。

---

## 3. Gather / Scatter — 非连续的内存访问

### 3.1 概念

Gather:  根据索引数组从源数组中收集数据

$$
\text{output}[i] = \text{input}[\text{index}[i]]
$$

Scatter: 根据索引数组向目标数组分发数据

$$
\text{output}[\text{index}[i]] = \text{input}[i]
$$

应用:
  - Embedding lookup (gather)
  - Attention 的 KV cache 更新 (scatter)
  - Sparse matrix operations

### 3.2 Triton 实现：Embedding Lookup (Gather)

```python
@triton.jit
def gather_kernel(input_ptr, index_ptr, output_ptr,
                  N, EMBED_DIM,
                  BLOCK_SIZE: tl.constexpr):
    """
    Embedding lookup: output[i, :] = input[index[i], :]
    
    Gather 在 GPU 上容易实现但性能差:
    - index 可以指向 input 的任意位置
    - 无法做 coalescing
    - L2 cache 命中率取决于 index 的分布（随机 → 低，顺序 → 高）
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # 加载索引
    indices = tl.load(index_ptr + offsets, mask=mask, other=0)
    
    # Gather: 根据索引从 input 加载
    # 每个线程访问不同的 input 行 → 非合并访问 → 慢
    for d in range(0, EMBED_DIM, BLOCK_SIZE):
        d_offsets = d + tl.arange(0, BLOCK_SIZE)
        d_mask = d_offsets < EMBED_DIM
        
        # 计算每个线程要访问的地址
        ptrs = input_ptr + indices[:, None] * EMBED_DIM + d_offsets[None, :]
        vals = tl.load(ptrs, mask=mask[:, None] & d_mask[None, :])
        
        out_ptrs = output_ptr + offsets[:, None] * EMBED_DIM + d_offsets[None, :]
        tl.store(out_ptrs, vals, mask=mask[:, None] & d_mask[None, :])

# Performance note:
# - 随机 gather: ~10-20% HBM bandwidth（最坏情况）
# - 顺序 gather: ~80% HBM bandwidth（接近 coalesced）
```

### 3.3 Triton 实现：Scatter

```python
@triton.jit
def scatter_kernel(input_ptr, index_ptr, output_ptr,
                   N, BLOCK_SIZE: tl.constexpr):
    """
    Scatter: output[index[i]] = input[i]
    
    问题: 多个线程可能写同一个 output 位置 → 需要 atomic
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    indices = tl.load(index_ptr + offsets, mask=mask, other=0)
    
    # Scatter with atomic add（如果可能有冲突）
    out_ptrs = output_ptr + indices
    tl.atomic_add(out_ptrs, vals, mask=mask)
```

---

## 4. Convolution — 把卷积映射到 GEMM

### 4.1 三种实现策略

1. Direct convolution:
   直接写 conv 的 7 层嵌套循环
   优点: 简单，内存友好
   缺点: 难优化，cache 不友好

2. Im2col + GEMM:

   $$
   \text{im2col}:\ (C, H, W) \to (C \cdot K \cdot K,\ H_{\text{out}} \cdot W_{\text{out}})
   $$

   $$
   \text{GEMM}:\ C_{\text{out}} \times (C \cdot K \cdot K) \otimes (C \cdot K \cdot K) \times (H_{\text{out}} \cdot W_{\text{out}})
   $$

   优点: 可以用高度优化的 GEMM
   缺点: im2col 有内存膨胀（$K \cdot K \times$）

3. Winograd / FFT:
   用数学变换减少乘法次数
   优点: 对特定 kernel size 极致优化
   缺点: 复杂，有精度问题

### 4.2 Triton 中实现 Depthwise Conv

```python
# 见 phase2_compute/06_depthwise_conv.py
# 关键: 每个 program 处理一个输出 spatial tile + 一组 channels
# 沿 KH×KW 维做 reduction

# 性能: 通常 memory-bound（算术强度 = KH*KW / (KH*KW+4) ≈ 0.5-0.9 FLOP/byte）
```

---

## 5. Atomic Operations — 处理并发写

### 5.1 Triton 支持的 Atomic

```python
tl.atomic_add(ptr, val)     # 原子加法
tl.atomic_max(ptr, val)     # 原子最大值
tl.atomic_min(ptr, val)     # 原子最小值
tl.atomic_and(ptr, val)     # 原子按位与
tl.atomic_or(ptr, val)      # 原子按位或
tl.atomic_xor(ptr, val)     # 原子按位异或
tl.atomic_cas(ptr, cmp, val) # 原子比较并交换
tl.atomic_xchg(ptr, val)    # 原子交换
```

### 5.2 Atomic 的性能代价

当多个线程对同一地址做 atomic 操作时:
  1 个线程: 正常速度
  2-4 个线程: 轻微变慢
  32+ 个线程: 严重串行化 → 可能比非 atomic 版本慢 10-100×

避免大量 atomic 竞争的策略:
  1. 先做 local reduce，最后再 atomic 写一次
  2. 用 tiling 减少 atomic 的粒度
  3. 如果可以，用 shared memory buffer 先缓冲再写

---

## 6. 常见模式的决策树

```
你想实现什么？

├── Elementwise: x[i] = f(y[i], z[i])
│   → 最简单的 kernel，memory-bound
│   → 考虑: 是否可以和相邻 op 融合？

├── Reduce: result = reduce(x, dim=d)
│   → memory-bound, 用 tl.sum/tl.max(axis=d)
│   → 考虑: 是否可以融入 producer/consumer kernel？

├── GEMM: C = A @ B
│   → 通常 compute-bound (大尺寸)
│   → 用 shared memory + autotune + num_stages
│   → 考虑: split-K 如果 K 很大

├── Scan: output[i] = f(output[i-1], input[i])
│   → 有数据依赖，但 Blelloch 算法做到 O(log n) 深度
│   → Block 内: 用 tl.associative_scan (编译器自动展开 warp shuffle + smem)
│   → 跨 Block: 必须双 kernel（block sum 通过 global memory 传递）
│   → 是 Reduce 的"镜像问题" — fan-out vs fan-in

├── Gather/Scatter:
│   → 通常 memory-bound, 合并访问难
│   → 随机 gather: 性能受限，接受就好

└── Attention:
    → memory-bound (长序列) 或 compute-bound (短序列)
    → 用 Flash Attention tiling
    → 可叠加 causal mask, GQA
```

---

## 参考资料

- [Triton Official Tutorials](https://triton-lang.org/main/getting-started/tutorials/)
- [GPU Gems 2 — Parallel Prefix Sum (scan)](https://developer.nvidia.com/gpugems/gpugems2/part-iv-general-purpose-computation-gpus-primer/chapter-39-parallel-prefix-sum-scan-cuda)
- [CUTLASS — Implicit GEMM for Convolution](https://github.com/NVIDIA/cutlass/blob/main/media/docs/implicit_gemm_convolution.md)
