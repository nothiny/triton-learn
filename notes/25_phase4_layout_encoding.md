# 25 — Layout Encoding 完全指南

> Triton 编译器最独特的设计——5 种 Layout 类型、convert_layout 的真实代价、type-driven lowering 的精妙之处。
> 配合 `phase4_compiler/04_layout_system.py` 和 `05_convert_layout.py`。

---

## 0. 问题：编译器怎么知道"哪个线程处理哪些数据"？

你写的 Triton 代码：

```python
x = tl.load(ptr + offs, mask=mask)   # 加载 256 个元素
y = x * 2 + 1                        # 计算
tl.store(out + offs, y, mask=mask)  # 存回去
```

你**没有**写任何关于线程分配的逻辑——不像 CUDA 那样手动计算 `threadIdx.x` 映射。但 GPU 执行时需要知道：thread 0 处理元素 0 还是元素 128？warp 0 管哪些行？

**答案：编译器通过 Layout Encoding 系统自动决定这一切。**

```
tensor<256xf32, #blocked<{
  sizePerThread  = [1],
  threadsPerWarp = [32],
  warpsPerCTA    = [4],
  order          = [0]
}>>
```

这串神秘符号的含义：**256 个元素 = 4 warps × 32 threads/warp × 1 elem/thread**。

---

## 1. 五种 Layout 类型

### 1.1 BlockedEncodingAttr — 标准分块（90% 的 op 用这个）

```
#ttg.blocked<{
  sizePerThread  = [s0, s1, ...],   ← 每线程在每个维度持有多少元素
  threadsPerWarp = [t0, t1, ...],   ← warp 内线程在各维度的分布
  warpsPerCTA    = [w0, w1, ...],   ← CTA 内 warp 在各维度的分布
  order          = [o0, o1, ...]    ← 维度优先级（innermost first）
}>
```

**公式**：CTA 在各维度的总大小 = warpsPerCTA × threadsPerWarp × sizePerThread

**例 1：1D vector<128>**

```
#blocked<{
  sizePerThread  = [1],
  threadsPerWarp = [32],
  warpsPerCTA    = [4],
  order          = [0]
}>
```

分配：元素 0→thread 0 (warp 0), 1→thread 1 (warp 0), ..., 31→thread 31 (warp 0), 32→thread 0 (warp 1), ...

**例 2：2D tile<128, 64>（常见于 GEMM）**

```
#blocked<{
  sizePerThread  = [1, 4],
  threadsPerWarp = [2, 16],
  warpsPerCTA    = [4, 1],
  order          = [0, 1]
}>
```

可视化：

```
dim 0 (128 rows)
┌──────────────────────────────────────────┐
│ Warp 0: rows 0-31,  cols 0-63            │
│  Thread (0,0): rows 0,1  cols 0-3       │ ← 1×4 = 4 elements
│  Thread (0,1): rows 0,1  cols 4-7       │
│  ...                                     │
├──────────────────────────────────────────┤
│ Warp 1: rows 32-63, cols 0-63            │
├──────────────────────────────────────────┤
│ Warp 2: rows 64-95, cols 0-63            │
├──────────────────────────────────────────┤
│ Warp 3: rows 96-127, cols 0-63           │
└──────────────────────────────────────────┘
```

**`order` 参数的含义**：

```
order=[0, 1] → dim 0 是 innermost（row-major 风格）
  → thread (0,0) 和 thread (0,1) 共享同一组 rows
  → 适合 dim 0 连续的内存（A 矩阵的行）

order=[1, 0] → dim 1 是 innermost（column-major 风格）
  → 适合 dim 1 连续的内存（B 矩阵的列）
```

`order` 影响 coalescing：innermost 维度的相邻线程访问连续内存地址 → coalesced access。

### 1.2 SliceEncodingAttr — 规约的"切片"

```
#ttg.slice<{
  dim = 1,                        ← 沿哪个轴做 reduction
  parent = #blocked<{...}>        ← 父布局（规约前的数据布局）
}>
```

**产生场景**：`tl.sum(x, axis=1)`

```
输入: tensor<M×N, #blocked<...>>    ← 每个线程持有 N 维上的几个元素
规约沿 axis=1 → 输出: tensor<M, #slice<{dim=1, ...}>>
```

`#slice` 表示"原本沿 dim=1 分布的线程现在被折叠了"——它们通过 warp shuffle + shared memory 协作完成规约。

**关键代价**：`#slice` 几乎总是需要一次 `convert_layout` 才能被后续 elementwise op 使用。

### 1.3 MmaEncodingAttr — Tensor Core 输出

```
#ttg.mma<{
  versionMajor = 2,               ← Ampere=2
  versionMinor = 0,
  instrShape = [16, 8, 16]        ← MMA 指令形状 M×N×K
}>
```

**产生场景**：`tl.dot(a, b)` 的输出。

与 BlockedEncoding 的本质区别：
- BlockedEncoding：每个线程持有**连续**的几个元素
- MmaEncoding：每个线程持有**分散**的 fragment——Tensor Core 要求数据按 warp-level matrix layout 排列

**instrShape 决定性能**：
- A100 (SM80): `m16n8k16`（fp16/bf16）或 `m16n8k8`（fp32）
- H100 (SM90): `m16n8k32`（fp16，K 翻倍，效率更高）或 `m16n8k64`（fp8）

### 1.4 DotOperandEncodingAttr — MMA 输入

```
#ttg.dot_op<{
  opIdx = 0,                      ← 0=A 操作数, 1=B 操作数
  parent = #blocked<{...}>        ← 数据原来的布局
}>
```

**产生场景**：`tl.dot(a, b)` 的输入（如果可以原位转换）。

```
tl.dot(a, b):
  a: #dot_op<opIdx=0, ...>    ← A 操作数，K 维 innermost
  b: #dot_op<opIdx=1, ...>    ← B 操作数，K 维 innermost
  → 输出: #mma<...>
```

**为什么需要单独的操作数布局**？Tensor Core 的 A 和 B 需要不同的数据排列方式——不能直接用 `#blocked` 喂给 MMA 指令。

### 1.5 ScanEncodingAttr — 前缀和

```
#ttg.scan<{
  dim = 0,
  parent = #blocked<{...}>
}>
```

Triton 2.1+ 支持。用于 `tl.cumsum`、`tl.cumprod` 等 scan 操作。

### 1.6 速查：什么产生什么 Layout？

| Triton 代码 | 产生的 Layout | 说明 |
|-----------|-------------|------|
| `tl.load(ptr, mask)` | `#blocked` | 编译器自动选择参数 |
| `x + y`, `x * 2` | 继承输入的 `#blocked` | Elementwise 不改变 layout |
| `tl.sum(x, axis=1)` | `#slice` | 规约轴被"折叠" |
| `tl.dot(a, b)` 输出 | `#mma` | Tensor Core 布局 |
| `tl.dot` 输入 | `#dot_op` | 如果可以原位转换 |
| `tl.cumsum(x)` | `#scan` | 保留维度，增加依赖 |

---

## 2. convert_layout：隐形的性能杀手

### 2.1 什么时候发生？

当两个 op 的 layout 不匹配时，编译器自动插入：

```mlir
%a_converted = ttg.convert_layout %a : #blocked → #mma
%b_converted = ttg.convert_layout %b : #blocked → #mma
%c = tt.dot %a_converted, %b_converted
```

### 2.2 代价有多大？

| 转换方向 | 实现方式 | 延迟 |
|---------|---------|------|
| `blocked ↔ blocked`（同参数） | 免费（no-op） | 0 |
| `blocked ↔ blocked`（不同参数） | warp shuffle | ~5 cycles |
| `blocked ↔ dot_op` | shared memory round-trip | ~20-30 cycles |
| `blocked ↔ mma` | shared memory + barrier | ~20-30 cycles |
| `slice → blocked` | shared memory + barrier | ~20-30 cycles |

最贵的转换需要：**写入 shared memory → bar.sync（等待所有线程写完）→ 从 shared memory 读出来**。

### 2.3 多少算"太多"？

```
0-1 个 convert_layout：正常范围
2-3 个：需要关注
4+ 个：很可能有性能问题
10+ 个：严重影响性能
```

### 2.4 实例：为什么 naive LayerNorm 很慢

```
x = tl.load(...)                    // #blocked
mean = tl.sum(x, axis=1) / N       // #slice（第一次 layout 变化）
x_centered = x - mean[:, None]     // #blocked ← #slice（convert!）
var = tl.sum(x_centered², axis=1)  // #slice（第二次 convert!）
x_norm = x_centered / rstd[:, None] // #blocked ← #slice（第三次 convert!）
```

每次 `blocked ↔ slice` 切换 → shared memory round-trip + barrier。三次 convert = 三次同步开销。这就是为什么生产级 LayerNorm 实现要特别管理 layout。

### 2.5 如何减少 convert_layout？

1. **合并同 layout 的 op**：把 elementwise 放在一起
2. **集中做 reduce**：减少 blocked ↔ slice 切换
3. **使用 persistent layout**：如果数据生来就是某个 layout，尽量不要转换
4. **检查 TTGIR**：养成 dump 后 grep `convert_layout` 的习惯

---

## 3. Type-Driven Lowering：Triton 的精妙设计

### 3.1 传统编译器的做法

```
IR_v1:  addf %a, %b                ← 统一 IR
         ↓ lowering pass
IR_v2:  fadd.f32 %a, %b            ← 目标特定 IR
```

**问题**：每次 lowering 都要**重写 op**。如果有 5 层 IR，同一语义要定义 5 种 op。

### 3.2 Triton 的做法（Type-Driven）

```
TTIR:   %x = tt.load %ptr : tensor<256xf32>
         ↓ ConvertTritonToTritonGPU
TTGIR:  %x = tt.load %ptr : tensor<256xf32, #blocked<{...}>>
         ↑ 还是 tt.load！op 没变，TYPE 变了
```

**核心洞察**：大多数 lowering 不是在"换 op"，而是在"给 type 加信息"。

```
TTIR → TTGIR:    type 加了 #blocked/#mma/#slice
AccelerateMatmul: tt.dot → 替换为 MMA op（这是少数需要换 op 的）
TTGIR → LLVM:    layout 被展开为线程索引计算
```

**好处**：
- 减少 op 数量（不需要为每个层级定义 load 的变体）
- 优化更简单（同一个 op 可以跨 layout 做优化）
- 可读性更强（你能看到"这还是同一个 load，只是数据分布变了"）

> 🔧 **Compiler Perspective**：这类似于 type-preserving compilation——type 本身携带了 lowering 所需的信息。在传统编译器中，数据布局信息通常隐式存在于寻址模式中；Triton 把它显式化为 type attribute。

---

## 4. Layout 参数是如何选择的？

`ConvertTritonToTritonGPU` pass 决定 layout 参数。它考虑：

1. **tensor 的 shape**——决定各维度的总大小
2. **`num_warps`**——决定 warpsPerCTA
3. **`BLOCK_SIZE`**（constexpr）——决定线程需要覆盖的元素总数
4. **op 的类型**——dot 需要 `#dot_op`/`#mma`，reduce 需要 `#slice`，其他用 `#blocked`

编译器自动计算 `sizePerThread` 和 `threadsPerWarp` 使得所有元素被覆盖。

**你不直接控制 layout，但通过 `num_warps` 和 `BLOCK_SIZE` 间接影响。** 这就是为什么 autotune 搜这两个参数——不同的值导致不同的 layout → 不同的寄存器使用 → 不同的性能。

---

## 5. 实战：从 TTGIR 中提取 Layout 信息

```bash
# 生成 TTGIR
TRITON_KERNEL_DUMP=1 python my_kernel.py

# 找 layout 声明
grep -o '#blocked<{[^}]*}>' ~/.triton/cache/*.ttgir
grep -o '#mma<{[^}]*}>' ~/.triton/cache/*.ttgir

# 统计 convert_layout
grep -c 'convert_layout' ~/.triton/cache/*.ttgir
```

或使用分析工具：

```python
from phase4_compiler.27_ir_analysis_tools import IROpStats

stats = IROpStats.from_ir_text(ttgir_text)
print(f"Layout types: {stats.layout_types}")
print(f"convert_layout: {stats.num_convert_layout}")
```

---

## 6. 参考文件

| 主题 | 教程文件 |
|------|---------|
| Layout 系统深度 | `04_layout_system.py` |
| convert_layout 代价 | `05_convert_layout.py` |
| ttg dialect 参考 | `25_triton_ttg_dialect.py` |
| IR 分析工具 | `27_ir_analysis_tools.py` |

---

## 7. 总结

```
Layout Encoding = Triton 编译器的核心创新

5 种 Layout:
  #blocked  ← 90% 的 op，标准分块
  #slice    ← 规约操作，"折叠"一个维度
  #mma      ← tl.dot 输出，Tensor Core 布局
  #dot_op   ← tl.dot 输入，MMA 操作数
  #scan     ← 前缀和

convert_layout:
  blocked ↔ slice → shared memory + barrier（~30 cycles）
  过多 → 性能杀手
  目标 → 每次 dump 后检查数量

Type-Driven Lowering:
  TTIR → TTGIR 的关键：op 不变，TYPE 加 layout
  这是 Triton 区别于传统编译器最精妙的设计
```
