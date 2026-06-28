# 24 — MLIR 框架深度：从概念到 Triton 的 Dialect 设计

> 理解 MLIR 不是一种语言，而是构建编译器的"乐高积木"。Triton 的 `tt` 和 `ttg` dialect 就是搭在这个框架上的两块关键积木。
> 配合 `phase4_compiler/22-27` 教程系列。

---

## 0. 为什么需要理解 MLIR？

你已经在用 MLIR 了——每一次运行 Triton kernel，编译器都在生成 MLIR。理解 MLIR 意味着：

- **读懂 `.ttir`/`.ttgir` 文件** → 知道编译器在"想什么"
- **诊断性能问题** → 看到 `convert_layout` 太多就知道问题在哪
- **跟踪 lowering** → 理解 `tt.dot` 如何一步步变成 `mma.sync`

Triton 是你用的，MLIR 是 Triton 用的。理解后者让你从"会用"到"懂原理"。

---

## 1. MLIR 的五个核心抽象

MLIR 不是一个 IR，而是一个**构建 IR 的框架**。任何 MLIR 程序由五个核心概念组成：

### 1.1 Operation（操作）—— 计算的基本单元

```mlir
%z = arith.addf %x, %y : f32
│   │      │    │   │    │
│   │      │    │   │    └── 结果类型
│   │      │    │   └────── operand（引用 SSA 值 %y）
│   │      │    └────────── operand（引用 SSA 值 %x）
│   │      └──────────────── op 名（arith dialect 的 addf op）
│   └─────────────────────── 结果 SSA 值
└─────────────────────────── 等号 = 有 return value
```

```mlir
tt.store %ptr, %val, %mask : tensor<256x!tt.ptr<f32>>
│         │      │     │
│         └──────┴─────┴──── operands（三个运行时值）
└─────────────────────────── 没有 %result = 前缀 → void op（side effect）
```

每个 op 有：所属 dialect、op 名、operands（运行时值）、results（输出）、attributes（编译期常量）。

### 1.2 Type（类型）—— 值的种类

```mlir
// 标量类型（builtin dialect）
i32, f32, f16, f64, i1, index

// Tensor 类型
tensor<256xf32>              // 纯数据（TTIR）
tensor<256xf32,              // 带 layout encoding（TTGIR）
       #blocked<{...}>>

// 自定义类型（dialect-specific）
!tt.ptr<f32>                 // Triton 指针类型
!tt.ptr<tensor<256xf32>>     // 指向 tensor 的指针
```

**类型驱动 lowering**：Triton 最精妙的设计——TTIR→TTGIR 不是换 op，而是在 TYPE 上加 attribute。

```
TTIR:   tensor<256xf32>          ← 只知道是 256 个 f32
TTGIR:  tensor<256xf32,          ← 知道了线程如何分配
                #blocked<{
                  sizePerThread=[1],
                  threadsPerWarp=[32],
                  warpsPerCTA=[4],
                  order=[0]
                }>>
```

### 1.3 Attribute（属性）—— 编译期常量

```mlir
// op 上的 attribute
%c = arith.constant 256 : i32     // 256 是 attribute
%offsets = tt.make_range {start=0, end=256}  // start/end 是 attribute

// type 上的 attribute
!tt.ptr<f32> {tt.divisibility = 16 : i32}   // 16 字节对齐（优化用）

// dialect attribute（# 前缀）
#ttg.blocked<{...}>                 // layout encoding（最重要的 attribute）
#ttg.mma<{versionMajor=2, instrShape=[16,8,16]}>
```

| | Operand | Attribute |
|---|--------|-----------|
| 何时确定 | 运行时 | 编译期 |
| 语法 | `(%name : type)` | `{key = value}` 或 `#dialect.attr` |
| 能否变 | 是（不同输入） | 否（同一 kernel 不变） |

### 1.4 Dialect（方言）—— 领域的命名空间

Dialect = 一组 op + type + attribute 的集合，有独立的命名空间。

Triton 编译器涉及的 dialect：

| Dialect | 用途 | 示例 |
|---------|------|------|
| `arith` | 算术运算 | `addf`, `mulf`, `addi`, `cmpi`, `constant` |
| `scf` | 结构化控制流 | `scf.for`, `scf.if`, `scf.yield` |
| `cf` | 低级控制流 | `br`, `cond_br` |
| **`tt`** | **Triton IR** | `tt.load`, `tt.store`, `tt.dot`, `tt.reduce` |
| **`ttg`** | **Triton GPU IR** | `ttg.convert_layout`, `ttg.async_copy` |
| `nvvm` | NVIDIA 特定（LLVM 阶段） | `nvvm.mma.sync`, `nvvm.read.ptx.sreg` |

关键设计：**多 dialect 可以在同一模块中混用**。同一个函数中，`tt.load` 和 `arith.addf` 和 `scf.for` 共存。

### 1.5 Region & Block（结构）

```
module {                          ← 最外层容器
  tt.func public @kernel(...) {   ← 函数 = 一个 kernel 入口（一个 Region）
    ^bb0(%arg0: i32):             ← Block 标签（可省略）
      %0 = arith.constant ...     ← Operation
      %1 = arith.addi ...         ← Operation
      tt.return                   ← Terminator（必须！）
  }
}
```

关键规则：
- 每个 Region 至少一个 Block
- 每个 Block 以 Terminator 结尾（`tt.return`, `scf.yield`）
- Block 可以有参数（MLIR 不需要 phi node）
- SSA 值的作用域是 Region 级别

---

## 2. MLIR 文本格式速成

### 2.1 基础语法

```mlir
// 注释
module {                          // 顶层 module
  tt.func public @name(           // public = GPU kernel 必须可见
    %x: !tt.ptr<f32>,             // 函数参数
    %N: i32
  ) attributes {noinline = false} {
    %c = arith.constant 256 : i32           // 定义常量
    %pid = tt.get_program_id x : i32        // 读取 blockIdx
    %offsets = tt.make_range {start=0, end=256} : tensor<256xi32>
    %block_start = arith.muli %pid, %c : i32
    %0 = tt.splat %block_start : i32 -> tensor<256xi32>
    %1 = arith.addi %0, %offsets : tensor<256xi32>
    // ... 更多 op ...
    tt.return
  }
}
```

### 2.2 SSA 命名规则

- `%name`：显式命名（重要的值）
- `%0`, `%1`, `%2`：自动编号（中间值）
- 每个值只能定义一次
- 使用前必须先定义（支配关系）

### 2.3 手工构造 MLIR 的价值

理解 MLIR 文本格式 → 能读懂 `.ttir` → 能诊断编译器问题 → 能手工优化 IR。

---

## 3. Triton 的两大自研 Dialect

### 3.1 `tt` Dialect（Triton IR）—— "做什么运算"

| Op | Python | 语义 |
|----|--------|------|
| `tt.load` | `tl.load(ptr, mask=...)` | 从内存加载 tensor |
| `tt.store` | `tl.store(ptr, val, mask=...)` | 存储 tensor 到内存 |
| `tt.dot` | `tl.dot(a, b)` | ★ 矩阵乘（触发 MMA lowering） |
| `tt.reduce` | `tl.sum(x, axis=1)` | 规约 |
| `tt.broadcast` | `tl.broadcast_to(x, shape)` | 广播 |
| `tt.splat` | (隐式 scalar+tensor) | 标量→tensor |
| `tt.addptr` | (隐式 ptr+offs) | 指针算术 |
| `tt.make_range` | `tl.arange(0, N)` | 生成 [0,1,...,N-1] |
| `tt.get_program_id` | `tl.program_id(0)` | 读取 blockIdx |
| `tt.trans` | (隐式) | 转置 |

**Python → TTIR 映射精度**：

```python
x = tl.load(x_ptr + offs, mask=mask)
```
变成：
```mlir
%p = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
%a = tt.addptr %p, %offsets : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
%x = tt.load %a, %mask : tensor<256x!tt.ptr<f32>>
```

简单的 `ptr + offset` 变成了 `splat` + `addptr` 两个 op。但这给了编译器优化空间——可以合并冗余的 `splat`。

### 3.2 `ttg` Dialect（Triton GPU IR）—— "怎么做"

#### Layout Encoding Types（最核心）

| Layout | 含义 | 出现场景 |
|--------|------|---------|
| `#ttg.blocked<{...}>` | 标准分块布局 | 所有 elementwise op |
| `#ttg.slice<{dim, parent}>` | 规约切片布局 | tl.sum, tl.max |
| `#ttg.mma<{instrShape}>` | Tensor Core MMA 布局 | tl.dot 输出 |
| `#ttg.dot_op<{opIdx, parent}>` | MMA 操作数布局 | tl.dot 输入 |
| `#ttg.scan<{...}>` | Scan/前缀和布局 | tl.cumsum |

#### GPU-Specific Operations

| Op | 描述 | 代价 |
|----|------|------|
| `ttg.convert_layout` | 改变 tensor 的 layout | 可能 shared memory round-trip |
| `ttg.async_copy` | 异步 global→shared 拷贝 | 低（不阻塞线程） |
| `ttg.async_commit_group` | 标记 async copy group | 免费 |
| `ttg.async_wait` | 等待 async copy 完成 | 取决于数据是否到达 |
| `ttg.local_alloc` | 分配 shared memory buffer | 编译期（占用配额） |

#### Module-Level Attributes

```mlir
module attributes {
    ttg.num-ctas = 1 : i32,           // CTA 数量
    ttg.num-warps = 4 : i32,          // Warp 数量
    ttg.target = "cuda:90",           // GPU 架构（H100）
    ttg.threads-per-warp = 32 : i32   // 固定为 32
}
```

> 🔧 **Compiler Perspective**：`ttg` dialect 的设计体现了"type-driven lowering"。TTIR 中的 `tensor<256xf32>` 到 TTGIR 中变成 `tensor<256xf32, #blocked<...>>`——op 不变，type 变。这比传统编译器的"op rewriting"更简洁、更容易优化。

---

## 4. MLIR Pass 基础设施

### 4.1 Pass 的类型

| 类型 | 作用 | Triton 中的例子 |
|------|------|---------------|
| **Conversion Pass** | Dialect A → Dialect B 的 lowering | `ConvertTritonToTritonGPU` |
| **Transform Pass** | 同一 dialect 内的优化 | `TritonGPURemoveLayoutConversions` |

### 4.2 Pattern Rewriting（最核心的 Pass 模式）

```
定义: pattern → replacement
引擎: 在 IR 中匹配 pattern，替换为 replacement
迭代: 反复应用直到没有新匹配（fixed-point）
```

**例：TritonCombineOps**

Pattern（2 条连续 elementwise）：
```mlir
%t = arith.mulf %x, %c1 : f32
%z = arith.addf %t, %c2 : f32
```

Replacement（融合为 1 条 FMA）：
```mlir
%z = arith.math.fma %x, %c1, %c2 : f32
```

**例：RemoveLayoutConversions**

Pattern（blocked → mma → blocked 的冗余转换）：
```mlir
%b : tensor<..., #mma> = ttg.convert_layout %a : #blocked
%c : tensor<..., #blocked> = ttg.convert_layout %b : #mma
```

Replacement：
```mlir
%c = %a   // 直接跳过中间的 mma 转换（如果 blocked 参数兼容）
```

### 4.3 Triton Pass Pipeline（完整顺序）

```
1. TTIR Optimization
   TritonInliner → TritonCombineOps → CanonicalizeOps → LoopUnroll

2. TTIR → TTGIR Conversion ★★★
   ConvertTritonToTritonGPU（分配 layout encoding）

3. TTGIR Optimization
   Coalesce → RemoveLayoutConversions → ★ AccelerateMatmul →
   CombineTensorSelect → OptimizeDotOperands

4. Software Pipelining
   ★ Pipeline（展开循环 + cp.async） → Prefetch

5. TTGIR → LLVM
   ConvertTritonGPUToLLVM（展开 layout，生成 NVVM intrinsic）

6. LLVM → PTX（LLVM 内置，Triton 不控制）
   LLVM Opt Pipeline → NVPTX CodeGen → Register Allocation → PTX
```

---

## 5. 实战：IR 分析工具箱

### 5.1 分析单个 IR 文件

```python
from phase4_compiler.27_ir_analysis_tools import IROpStats

stats = IROpStats.from_ir_text(open("kernel.ttir").read())
print(f"Operations: {dict(stats.op_counts)}")
print(f"Dialects: {dict(stats.dialect_counts)}")
print(f"Convert layouts: {stats.num_convert_layout}")
print(f"Registers: ~{stats.register_estimate}")
```

### 5.2 对比不同 Config

```python
from phase4_compiler.27_ir_analysis_tools import IRComparator

stats_a = IROpStats.from_ir_text(ptx_a)
stats_b = IROpStats.from_ir_text(ptx_b)
print(IRComparator.compare(stats_a, stats_b, "num_warps=4", "num_warps=8"))
# → 显示寄存器数变化、convert_layout 差异等
```

### 5.3 检查清单

| 问题 | 检查什么 | IR 层级 |
|------|---------|--------|
| tl.dot 被识别了吗？ | `tt.dot` 存在于 TTIR | `.ttir` |
| convert_layout 有多少？ | `ttg.convert_layout` 计数 | `.ttgir` |
| MMA 触发了没有？ | `mma.sync` in PTX | `.ptx` |
| 寄存器 spill 了吗？ | `.reg` 声明数量 > 200? | `.ptx` |
| Shared memory 够吗？ | `.shared` 声明总大小 | `.ptx` |

---

## 6. 总结

```
MLIR 框架
  └── Triton 编译器
        ├── tt dialect（"做什么运算"——纯数学描述）
        │     └── 操作: load, store, dot, reduce, ...
        └── ttg dialect（"怎么做"——GPU 线程分配）
              ├── 类型: #blocked, #mma, #slice, #dot_op
              └── 操作: convert_layout, async_copy, ...

  关键设计:
    • Type-driven lowering — 不是换 op，是在 type 上加 layout
    • Multi-dialect — arith + scf + tt + ttg 在同一模块中
    • Pattern Rewriting — 声明式优化，反复迭代到不动点
    • 渐进 lowering — 每个 pass 只改变一个维度
```

> 🔧 **记于心：MLIR 不是"一种 IR"，而是一个 IR 框架。Triton 的 `tt` 和 `ttg` 是两个 plugins。理解这个框架，你就能理解 Triton 为什么会这样设计编译器。**

---

## 7. 参考文件

| 主题 | 教程文件 |
|------|---------|
| MLIR 核心概念 | `22_mlir_core_concepts.py` |
| MLIR 文本格式 | `23_mlir_text_format.py` |
| `tt` dialect 参考 | `24_triton_tt_dialect.py` |
| `ttg` dialect 参考 | `25_triton_ttg_dialect.py` |
| Pass 系统 | `26_mlir_pass_system.py` |
| IR 分析工具 | `27_ir_analysis_tools.py` |
| 编译器管线基础 | `01-13` 教程系列 |
| 编译器内部深度 | `14-21` 教程系列 |
