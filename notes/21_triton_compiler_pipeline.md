# 21 — Triton 编译器管线

> ⭐ **这是整个项目最核心的笔记。** 理解 Triton 编译器如何把你的 Python kernel 变成 GPU 指令，是写出高性能 kernel 的关键。
> 前半部分写给所有人，后半部分（🔧 标记）写给有编译器背景的读者。

---

## 0. 前言：为什么你需要了解编译器内部？

### 0.1 一个让你"啊哈"的问题

你写了这段 Triton 代码：

```python
x = tl.load(ptr + offsets, mask=mask)      # 加载
y = x * 2 + 1                              # 计算
tl.store(out_ptr + offsets, y, mask=mask)  # 存储
```

问题：**编译器怎么决定 1024 个元素由哪些线程处理？每个线程处理几个？线程访问内存的顺序是什么？**

答案是：编译器通过一个叫做 **Layout Encoding** 的系统来决定这一切。`#blocked<{sizePerThread=[1], threadsPerWarp=[32], warpsPerCTA=[4], order=[0]}>` 这串神秘符号，就是把"一个 block 处理 1024 个元素"翻译成"4 个 warp × 每个 warp 32 个线程 × 每个线程 1 个元素"的具体方案。

理解这个过程，你就能：
- 读懂生成的 PTX，判断编译器有没有做你想要的优化
- 理解为什么有些代码写法快、有些慢（layout conversion 的代价）
- 在性能出问题时，知道从哪里下手排查

### 0.2 什么是 IR（中间表示）？

如果你不熟悉编译器：

```
IR (Intermediate Representation) = 编译器的"中间语言"

源代码 → IR_1 → IR_2 → IR_3 → 机器码

每一层 IR 丢掉一些高层信息，加入一些底层细节。
就像翻译: 中文 → 世界语 → 英语 → 二进制摩斯电码
  - 中文: "把门打开"（最接近你的想法）
  - 世界语: "malfermi la pordon"（去掉了中文的语法细节）
  - 英语: "open the door"（更接近西方语言习惯）
  - 摩斯电码: --- .--. . -. / - .... . / -.. --- --- .-.（机器能懂的）
```

Triton 编译器的 IR 有四层：

| 阶段 | 语言 | 通俗类比 | 包含了什么 |
|------|------|---------|-----------|
| Python | `@triton.jit` 函数 | 你的源代码 | "对每个 block，加载一些数据，加一下，写回去" |
| TTIR | MLIR (tt dialect) | 通用的"数学描述" | "加载 tensor<1024xf32>，做 addf，存回去" |
| TTGIR | MLIR (ttg dialect) | 加上了"怎么分配线程" | "tensor<1024xf32> 按 blocked layout 分布到 4 个 warp" |
| LLVM IR | LLVM IR (NVPTX) | 接近机器码的通用表示 | 寄存器分配、内存地址计算、分支 |
| PTX | NVIDIA GPU 汇编 | 真正给 GPU 看的指令 | `ld.global.f32`, `add.f32`, `st.global.f32` |

---

## 1. IR 层级全景图

```
你写的 Python (@triton.jit)
    │
    │  Triton 解析 Python AST
    ▼
TTIR  (tt dialect, MLIR)               ← 通用 Triton IR，还不知道 GPU 的事
    │                                      "这个 kernel 做了哪些 tensor 运算？"
    │  ConvertTritonToTritonGPU
    │  （最关键的一步！）
    ▼
TTGIR (tt + ttg dialect, MLIR)         ← 加入 layout encoding
    │                                     "每个线程处理哪些数据？怎么分布？"
    │  AccelerateMatmul, Pipeline,
    │  Prefetch, Coalesce, ...
    ▼
LLVM IR                                 ← 标准 LLVM IR（NVPTX target）
    │                                     "寄存器怎么分配？地址怎么算？"
    │  LLVM NVPTX backend
    ▼
PTX                                     ← NVIDIA GPU 汇编
    │                                     "ld.global.f32, add.f32, st.global.f32"
    │  ptxas (NVIDIA 汇编器)
    ▼
SASS / CUBIN                            ← 最终执行的二进制
    │                                     000101011100...
```

> 🔧 **Compiler Perspective**: 这类似于 LLVM 的 dialect lowering：`Lang → Generic MLIR → Target MLIR → LLVM IR → MC`。TTIR→TTGIR 的转换是最关键步骤——它决定数据到线程的映射，类似 polyhedral compiler 中从 statement 到 loop nest + data mapping 的转换。

---

## 2. 从一个简单 kernel 的编译过程说起

让我们追踪 `vector_add` 这个简单 kernel 经过每一层 IR 的样子。

### 2.1 源代码（Python）

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)
```

### 2.2 TTIR（Triton IR）— 纯数学描述

TTIR 只知道"有 tensor 运算"，不知道 GPU 的任何细节。

```mlir
// TTIR 示例 — 关注"做了什么运算"
func.func @add_kernel(%x: !tt.ptr<f32>, %y: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32) {
  // 获取 block 索引
  %pid = tt.get_program_id {axis = 0 : i32}
  
  // 生成 [0, 1, 2, ..., BLOCK_SIZE-1]
  %offsets = tt.make_range {start = 0, end = 1024 : i32}
  
  // 计算全局偏移 = pid * BLOCK_SIZE + offsets
  %block_start = arith.muli %pid, %c1024_i32
  %global_offsets = arith.addi %block_start, %offsets
  
  // 边界检查
  %mask = arith.cmpi slt, %global_offsets, %n
  
  // 加载 x 和 y
  %x_vals = tt.load %x[%global_offsets] : tensor<1024xf32>, %mask
  %y_vals = tt.load %y[%global_offsets] : tensor<1024xf32>, %mask
  
  // 计算
  %result = arith.addf %x_vals, %y_vals : tensor<1024xf32>
  
  // 存储
  tt.store %out[%global_offsets], %result, %mask
  return
}
```

关键观察：
- 只有 **tensor 类型**，没有 thread/warp 信息
- `tt.load` / `tt.store` 仍然是对整个 tensor 的操作
- 没有任何 GPU 硬件相关的概念

### 2.3 TTGIR（Triton GPU IR）— 加入线程映射

经过 `ConvertTritonToTritonGPU` 后，**每个 tensor 都带上了 layout encoding**：

```mlir
// TTGIR 示例 — 关注新出现的 #blocked<...>
%x_vals = tt.load %x[%global_offsets]
    : tensor<1024xf32,      ← 原来是纯 tensor
             #blocked<{      ← 现在带上了 layout 信息！
               sizePerThread  = [1],       ← 每个线程持有 1 个元素
               threadsPerWarp = [32],      ← 每个 warp 32 个线程沿 dim 0
               warpsPerCTA    = [4],       ← 4 个 warp
               order          = [0]        ← dim 0 为 innermost（连续）
             }>>

// 解读: 1024 个元素 = 4 warps × 32 threads/warp × 1 element/thread
//       每个线程处理 1 个元素，线程 0 处理第 0 个，线程 1 处理第 1 个...
```

这就是 Layout Encoding 的核心——**描述如何将逻辑上的 tensor 元素分配到物理上的线程**。

### 2.4 LLVM IR — 通用底层表示

TTGIR 经过 `ConvertTritonGPUToLLVM` 后变成标准 LLVM IR：

```llvm
; LLVM IR 示例 — 关注寄存器、地址计算
define void @add_kernel(ptr %x, ptr %y, ptr %out, i32 %n) {
  %pid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()  ; blockIdx.x
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()     ; threadIdx.x
  
  %offset = add i32 %pid_mul, %tid
  %addr_x = getelementptr float, ptr %x, i32 %offset
  %addr_y = getelementptr float, ptr %y, i32 %offset
  
  %val_x = load float, ptr %addr_x
  %val_y = load float, ptr %addr_y
  %result = fadd float %val_x, %val_y
  
  %addr_out = getelementptr float, ptr %out, i32 %offset
  store float %result, ptr %addr_out
  ret void
}
```

关键观察：
- 出现了具体的 **寄存器** 和 **地址计算**
- `threadIdx.x` 被显式读取（之前由 `sizePerThread` 隐式描述）
- 但 LLVM IR 仍然是 **target-independent** 的（不直接包含 PTX 指令）

### 2.5 PTX — GPU 汇编

LLVM NVPTX backend 生成 PTX：

```asm
// PTX 示例 — 这是真正在 GPU 上执行的指令
.visible .entry add_kernel(
    .param .u64 add_kernel_param_0,
    .param .u64 add_kernel_param_1,
    .param .u64 add_kernel_param_2,
    .param .u32 add_kernel_param_3
)
{
    .reg .f32   %f<5>;          // 寄存器声明
    .reg .b32   %r<10>;
    .reg .b64   %rd<10>;

    // 读取 blockIdx.x 和 threadIdx.x
    mov.u32     %r1, %ctaid.x;      // pid
    mov.u32     %r2, %tid.x;        // thread id within block
    // ...
    
    // 从 HBM 加载数据
    ld.global.f32   %f1, [%rd2];    // [LD.GLOBAL] 加载 (~500 cycles)
    ld.global.f32   %f2, [%rd3];
    
    // 计算
    add.f32         %f3, %f1, %f2;
    
    // 写回 HBM
    st.global.f32   [%rd4], %f3;    // [ST.GLOBAL] 存储
    ret;
}
```

关键观察：
- `.reg` 声明告诉你有多少寄存器（寄存器压力的直接指标）
- `ld.global` / `st.global` = HBM 访问（慢！）
- 这就是 `phase3_compiler/04_ptx_analysis.py` 分析的内容

---

## 3. Layout Encoding 系统 — 编译器如何决定"哪个线程处理哪个数据"

这是 Triton 编译器最核心的设计。

### 3.1 四种 Layout 类型

| Layout | 含义 | 何时出现 |
|--------|------|---------|
| `BlockedEncodingAttr` | "标准分块"：沿各维等分给 warps/threads | 大多数 elementwise op 的输出 |
| `SliceEncodingAttr` | "切片"：沿某一维做 reduction | reduction 操作（sum, max）的中间结果 |
| `MmaEncodingAttr` | "MMA 布局"：Tensor Core 需要的特殊数据排列 | `tl.dot` 的输出 |
| `DotOperandEncodingAttr` | "MMA 操作数布局"：Tensor Core 输入的特殊排列 | `tl.dot` 的输入 |

### 3.2 BlockedEncodingAttr 详解

```
#blocked<{
  sizePerThread  = [1, 4],     ← 每个线程持有 dim0×1, dim1×4 个连续元素
  threadsPerWarp = [2, 16],    ← warp 内 2×16=32 个线程
  warpsPerCTA    = [4, 1],     ← 4 个 warp 沿 dim0
  order          = [0, 1]      ← dim0 为 innermost（连续维）
}>

可视化（对于 128×64 的 tensor）:
  
  每个 CTA（block）处理 128×64:
  
  dim 0 (128 rows)
  ┌───────────────────────────────────┐
  │ Warp 0: rows 0-31, cols 0-63      │  threadsPerWarp=[2,16]
  │   Thread (0,0):  rows 0,1  cols 0-3   │  sizePerThread=[1,4]
  │   Thread (0,1):  rows 0,1  cols 4-7   │
  │   ...                                │
  ├───────────────────────────────────┤
  │ Warp 1: rows 32-63, cols 0-63     │
  ├───────────────────────────────────┤
  │ Warp 2: rows 64-95, cols 0-63     │
  ├───────────────────────────────────┤
  │ Warp 3: rows 96-127, cols 0-63    │
  └───────────────────────────────────┘

  元素总数 = (4 warps × 2×16 threads × 1×4 elements) = 128×64 ✓
```

### 3.3 怎么读 Layout？


$$
\begin{aligned}
\text{公式: total\_size} &= (\text{warpsPerCTA} \times \text{threadsPerWarp} \times \text{sizePerThread}) \\[4pt]
\text{逐个维度:} \\
\text{dim 0: } &\text{warpsPerCTA}[0] \times \text{threadsPerWarp}[0] \times \text{sizePerThread}[0] \\
&= 4 \times 2 \times 1 = 8 \quad (\text{等等...不应该是 128 吗？})
\end{aligned}
$$

注意: 上面的是每个 warp 内部的分布，乘以 warpsPerCTA 才是 CTA 总数。

$$
\text{warpsPerCTA} \times \text{threadsPerWarp} \times \text{sizePerThread} = 4 \times 2 \times 1 \times 2 \times 16 \times 4
$$

但各维是交叉的，要按 order 去理解。

$$
\begin{aligned}
\text{简化理解:} \\
\text{dim 0 的分布: } &\text{warpsPerCTA}[0] \times \text{threadsPerWarp}[0] \times \text{sizePerThread}[0] = 4 \times 2 \times 1 = 8 \\
\text{dim 1 的分布: } &\text{warpsPerCTA}[1] \times \text{threadsPerWarp}[1] \times \text{sizePerThread}[1] = 1 \times 16 \times 4 = 64
\end{aligned}
$$

等等，$8 \times 64 \neq 128 \times 64$...实际上 `order=[0,1]` 意味着 dim0 是 innermost，各维度的乘法关系是: `threadsPerWarp` 的各维乘积 $= 32$，`warpsPerCTA` 的各维 $\approx$ `num_warps`.


> 💡 **实际建议**: 不需要能心算 layout。用 `phase3_compiler/02_layout_analysis.py` 可视化。理解概念就行：layout 就是"线程→数据"的分配方案。

### 3.4 `ConvertLayout` op — 潜在的性能杀手

当两个 op 的 layout 不匹配时，编译器插入隐式转换：

```mlir
// blocked layout → MMA layout（例如: elementwise op 的结果给 tl.dot）
%converted = ttg.convert_layout %val
    : tensor<128x64xf16, #blocked> → tensor<128x64xf16, #mma>
```

这个转换的代价：
- **可能需要 shared memory round-trip**（写到 shared memory，读出来，重排）
- **可能需要 barrier 同步**（等所有线程写完再读）
- 过多 `ConvertLayout` 是 Triton 性能差的主要原因之一

```
可视化 convert_layout:

Blocked layout:                    MMA layout:
  Thread 0: [a0, a1, a2, a3]       Thread 0: [a0, a4, b0, b4]
  Thread 1: [a4, a5, a6, a7]       Thread 1: [a1, a5, b1, b5]
  Thread 2: [b0, b1, b2, b3]  →   Thread 2: [a2, a6, b2, b6]
  Thread 3: [b4, b5, b6, b7]       Thread 3: [a3, a7, b3, b7]

数据需要重新分配 → 写到 shared memory → 读出来重排 → 额外的延迟
```

> 🔧 **Compiler Perspective**: `ConvertLayout` 类似寄存器分配中的 spill/reload，但粒度是 warp/block 级别。Triton 的 `TritonGPURemoveLayoutConversions` pass 尝试消除冗余转换（类似 copy propagation）。

---

## 4. 关键 Pass 详解

### 4.1 Pass 总览

| Pass | 做什么 | 通俗类比 |
|------|--------|---------|
| `TritonInliner` | 内联函数调用 | 把"调用函数"替换成"函数体" |
| `TritonCombineOps` | 融合相邻 elementwise op | 把 `add + relu` 合并为一个操作 |
| **`ConvertTritonToTritonGPU`** | TTIR → TTGIR，插入 layout | **最关键！** 决定数据→线程的映射 |
| `TritonGPUCoalesce` | 合并相邻内存访问 | 把 32 个独立 load 合并成一个 coalesced load |
| `TritonGPUAccelerateMatmul` | `tl.dot` → MMA intrinsic | 把"矩阵乘"翻译成 Tensor Core 指令 |
| `TritonGPUPipeline` | Software pipelining | 让加载和计算重叠执行 |
| `TritonGPUPrefetch` | 插入 prefetch 指令 | 提前把下一轮数据搬到 shared memory |
| `TritonGPURemoveLayoutConversions` | 消除冗余 layout 转换 | 去掉不必要的"数据重排" |
| `ConvertTritonGPUToLLVM` | TTGIR → LLVM IR | 最终 lowering 到 LLVM |

### 4.2 最重要的三个 Pass

**1. `ConvertTritonToTritonGPU` — 从抽象到具体**

这是整个编译流程中最关键的一步。

输入: TTIR（纯数学描述，无 GPU 信息）
输出: TTGIR（每个 op 有 layout encoding，知道数据怎么分配到线程）

决策内容:
  - 每个 tensor 的 layout（blocked/mma/slice/dot_operand）
  - 每个 op 的 thread/warp 分配
  - 插入 ConvertLayout 来处理 layout 不兼容

类比: 这是寄存器分配 + 指令调度 + 数据布局的融合 pass。
      在传统编译器中，这些是分开做的；Triton 一次性做完。

**2. `TritonGPUAccelerateMatmul` — 启用 Tensor Core**


$$
`tt.dot` → 识别为矩阵乘 → 替换为 MMA intrinsic

- 输入: `tt.dot(%a, %b) : tensor⟨M×K⟩ ⊗ tensor⟨K×N⟩ → tensor⟨M×N⟩`
- 输出: MMA 操作序列，使用 MmaEncodingAttr，自动选择 m16n8k16 (Ampere) 或 m16n8k32 (Hopper)
- 类比: LLVM 的 ISel（指令选择）。把"通用运算"替换为"硬件特定指令"
$$


**3. `TritonGPUPipeline` — Software Pipelining**

```
num_stages=1:
  |-- load[1] --|-- compute[1]--|-- load[2] --|-- compute[2] --|

num_stages=2:
  |-- load[1] --|-- load[2] --|-- load[3] --|
    |               |-- compute[1] --|-- compute[2] --|-- compute[3] --|
  （加载和计算重叠）

实现方式:
  编译器展开 K 维循环 → 
  插入 cp.async（异步拷贝指令）→
  插入 prefetch →
  管理 buffer 切换

类比: VLIW 编译器中的 modulo scheduling。
      展开循环 → 建立数据依赖图 → 计算 initiation interval → 重排指令。
```

---

## 5. 寄存器分配：Triton 的间接控制

### 5.1 Triton 不做寄存器分配

重要事实：Triton 编译器**不直接做寄存器分配**。这一工作交给 LLVM NVPTX backend。

但 Triton 通过以下参数**间接控制**寄存器压力：

| 参数 | 如何影响寄存器 | 效果 |
|------|--------------|------|
| `num_warps` ↑ | 更多 warp → 每个 warp 可用的寄存器更少 | 可能触发 LLVM spill |
| `sizePerThread` ↑ | 每线程持有更多元素 → 需要更多寄存器 | 直接增加寄存器需求 |
| `num_stages` ↑ | 更多 pipeline buffer → 需要更多 shared memory | 间接影响（shared memory 挤压 occupancy） |

### 5.2 三资源约束

每个 SM 的资源是固定的（H100）:

```
┌──────────────────┬───────────┐
│ 资源              │ 总量       │
├──────────────────┼───────────┤
│ 寄存器            │ 65536     │
│ Shared Memory     │ 228 KB    │
│ 最大 Warps        │ 64        │
│ 最大 Blocks       │ 32        │
└──────────────────┴───────────┘

```
你的 kernel 需要:
- 寄存器: $\text{num\_warps} \times 32 \times \text{registers\_per\_thread}$
- Shared Memory: $\text{num\_stages} \times \text{tile\_size} \times \text{dtype\_size}$
- Warps: $\text{num\_warps}$

如果任何一项超限，occupancy 就会下降。
Triton 的 autotuner 就是在搜索这个三维空间。

> 🔧 **Compiler Perspective**: 这是经典的多目标资源优化问题。传统 CPU 编译器只需要考虑寄存器（spill cost），但 GPU 编译器需要同时优化 registers、shared memory、occupancy 三个相互制约的目标。Triton 的 autotuner 本质上是 iterative compilation——尝试多种配置，测实际性能，选最优的。

---

## 6. 实战：如何 Dump IR 进行分析

### 6.1 三种 dump 方法

```bash
# 方法 1: 环境变量（最简单）
TRITON_KERNEL_DUMP=1 python my_kernel.py
# 输出写入 ~/.triton/cache/

# 方法 2: 看每个 MLIR pass 后的变化
MLIR_PRINT_IR_AFTER_ALL=1 python my_kernel.py
# 大量输出，适合深入 debug

# 方法 3: 只看特定 pass
TRITON_KERNEL_DUMP=1 python phase3_compiler/01_dump_ir.py
```

### 6.2 Dump 后看什么？

在 `~/.triton/cache/` 中找到文件后：

| 文件后缀 | 关注什么 | 问题诊断 |
|---------|---------|---------|
| `.ttir` | `tt.dot` 是否被识别？ | 如果 `tt.dot` 仍在，意味着 AccelerateMatmul 没触发 |
| `.ttgir` | `#blocked<...>` 参数是否合理？ | `sizePerThread` 太大 → 寄存器压力 |
| `.ttgir` | 有多少 `ttg.convert_layout`？ | 太多 → 性能杀手，尝试重构 kernel |
| `.ll` | 有没有 `alloca`（spill）？ | 有 spill → 寄存器压力大，调小 `num_warps` |
| `.ptx` | `.reg` 声明了多少寄存器？ | >128 per thread → 高寄存器压力 |
| `.ptx` | 有没有 `st.shared` + `bar.sync`？ | 多 → 可能有隐式的 layout conversion |

### 6.3 举个实际例子

```bash
# Dump tiled matmul 的 IR
TRITON_KERNEL_DUMP=1 python phase2_compute/02_matmul_tiled.py

# 在 ~/.triton/cache/ 中找到 .ptx 文件
# 搜索关键指令:
grep "mma.sync" ~/.triton/cache/*.ptx       # 是否用了 Tensor Core？
grep "ld.shared" ~/.triton/cache/*.ptx      # shared memory 加载多少？
grep "st.global" ~/.triton/cache/*.ptx      # HBM 写入次数
grep ".reg" ~/.triton/cache/*.ptx | wc -l  # 寄存器使用量
```

---

## 7. 总结：Triton 编译器的心智模型

```
       你的 Triton 代码
             │
    ┌────────┴────────┐
    │ Python AST 解析  │
    └────────┬────────┘
             │
    ┌────────┴────────┐
    │    TTIR (MLIR)   │  纯数学描述——做了哪些 tensor 运算
    └────────┬────────┘
             │
    ┌────────┴────────┐  ← 最关键的步骤
    │ Layout 分配      │  编译器决定: 数据→线程的映射
    │ ConvertLayout 插入│  编译器决定: layout 转换点
    └────────┬────────┘
             │
    ┌────────┴────────┐
    │   TTGIR (MLIR)   │  带 layout 的 tensor 运算
    │   ttg.convert    │
    │   ttg.async_copy │
    └────────┬────────┘
             │
    ┌────────┴────────┐
    │  MMA → 硬件指令  │  把 tl.dot 翻译成 Tensor Core 指令
    │  Pipeline 插入   │  把串行循环变成流水线
    │  Prefetch 插入   │  提前加载下一轮数据
    └────────┬────────┘
             │
    ┌────────┴────────┐
    │    LLVM IR       │  通用底层表示
    └────────┬────────┘
             │
    ┌────────┴────────┐
    │   LLVM Backend   │  寄存器分配（这里才做！）
    │   (NVPTX)        │  指令选择
    └────────┬────────┘
             │
             ▼
          PTX → SASS
```

**核心要点**:
1. Triton 编译器最重要的是 **layout encoding 系统** — 它决定了数据如何分配到线程
2. 你不直接控制线程，但通过 `BLOCK_SIZE`、`num_warps`、`num_stages` 等参数间接影响
3. Dump IR 是 debug 性能问题的最有效手段
4. `ConvertLayout` 是潜在的隐形性能杀手

---

## 7. 实战：写一个自定义 MLIR Pass（配合 `phase3_compiler/custom_pass/`）

### 7.1 为什么需要自定义 Pass？

Triton 的内置 passes 覆盖了大多数场景，但有时你需要:

1. 实验性优化: "如果我把这两个 op 融合，会快多少？"
2. 调试: 在特定 pass 前后注入 IR dump
3. 分析: 统计 kernel 中某种 op 的数量、shared memory 使用

Triton 提供了 Python API 来注册自定义 pass。

### 7.2 自定义 Pass 的基本框架

```python
# phase3_compiler/custom_pass/my_pass.py

from triton.compiler import passes

@passes.register_pass("my_custom_pass")
def my_pass(module):
    """
    一个最简单的自定义 pass: 打印 module 中所有的 op 名称。
    
    module: triton MLIR module (可以像 Python 对象一样遍历)
    """
    # 遍历 module 中的所有函数
    for func in module.body:
        if hasattr(func, 'body'):
            print(f"Function: {func.name}")
            # 遍历函数中的所有 op
            for op in func.body:
                print(f"  {op.name}: {op.operands}")

# 注册后，使用环境变量启用:
# TRITON_CUSTOM_PASSES=my_custom_pass python my_kernel.py
```

### 7.3 实际的简单 Pass: 统计 Shared Memory 使用

```python
@passes.register_pass("estimate_shared_mem")
def estimate_shared_mem_pass(module):
    """
    估计 kernel 的 shared memory 用量。
    
    遍历所有 ttg.convert_layout op（这些可能导致 shared memory round-trip），
    估算每个 convert 需要的 shared memory 字节数。
    """
    total_shared_bytes = 0
    
    for func in module.body:
        for block in func.body:
            for op in block:
                # 找 load 操作（可能产生 shared memory staging）
                if "load" in str(op.name):
                    result_type = op.result.type
                    if hasattr(result_type, 'shape'):
                        # 估算 tile 大小
                        shape = result_type.shape
                        dtype_size = 2  # 假设 fp16
                        tile_bytes = 1
                        for dim in shape:
                            tile_bytes *= dim
                        tile_bytes *= dtype_size
                        total_shared_bytes += tile_bytes
                        print(f"  [{op.name}] tile: {shape}, "
                              f"approx {tile_bytes} bytes")
    
    print(f"\n[Shared Memory Estimate] ~{total_shared_bytes / 1024:.1f} KB per block")
```

### 7.4 更高级: 分析 Kernel 的 Arithmetic Intensity

```python
@passes.register_pass("analyze_arith_intensity")
def analyze_arith_intensity(module):
    """
    统计 kernel 的 FLOPs 和 HBM 访问量，计算算术强度。
    
    简单实现: 统计 load/store 数量（HBM 访问）和 dot（compute）。
    """
    num_loads = 0
    num_stores = 0
    num_mma_ops = 0
    
    def count_ops(op):
        nonlocal num_loads, num_stores, num_mma_ops
        name = str(op.name)
        if "load" in name:
            num_loads += 1
        elif "store" in name:
            num_stores += 1
        elif "dot" in name or "mma" in name:
            num_mma_ops += 1
    
    for func in module.body:
        for block in func.body:
            for op in block:
                count_ops(op)
    
    total_hbm_bytes = (num_loads + num_stores) * 128 * 64 * 2  # rough estimate
    total_flops = num_mma_ops * 2 * 16 * 8 * 16  # 2×M×N×K per MMA
    
    ai = total_flops / total_hbm_bytes if total_hbm_bytes > 0 else float('inf')
    
    print(f"Loads: {num_loads}, Stores: {num_stores}, MMA ops: {num_mma_ops}")
    print(f"Estimated FLOPs: {total_flops:,}")
    print(f"Estimated HBM bytes: {total_hbm_bytes:,}")
    print(f"Arithmetic Intensity: {ai:.1f} FLOP/byte")
```

### 7.5 如何运行自定义 Pass

```bash
# 方法 1: 环境变量
TRITON_CUSTOM_PASSES=my_custom_pass,estimate_shared_mem python my_kernel.py

# 方法 2: 在代码中注册
# 只需 import 你的 pass 文件，pass 会被自动注册
import phase3_compiler.custom_pass.my_pass
# 然后正常运行 kernel

# 方法 3: 配合 IR dump 使用
TRITON_KERNEL_DUMP=1 TRITON_CUSTOM_PASSES=analyze_arith_intensity python my_kernel.py
# 可以同时看到 IR dump 和你的 pass 输出
```

### 7.6 Triton Pass 系统的限制

当前 (Triton 3.x) 的限制:

1. Python pass API 仍在发展中
   → 不是所有 triton.compiler.passes 都暴露给 Python
   → 复杂的 pass 可能需要 C++ 实现

2. 不能修改 IR 结构（只读分析是安全的）
   → Python pass 主要用于分析和 debug
   → 修改 IR 需要深入了解 Triton 的 MLIR 绑定

3. Pass 顺序是固定的
   → 你不能重新排序 passes
   → 自定义 pass 被插入到固定位置

对于真正的 pass 开发:
  → 需要 fork triton 源码
  → 用 C++ 写 MLIR pass（lib/Conversion/ 下）
  → 重新编译 triton

---

## 参考资料

- [MLIR Documentation](https://mlir.llvm.org/docs/)
- [Triton 编译器论文 (Tillet et al., 2023)](https://www.eecs.harvard.edu/~htk/publication/2023-pact-tillet-kung-cox.pdf)
- [Triton 源码 — lib/Conversion/](https://github.com/triton-lang/triton/tree/main/lib/Conversion)
- [Triton 源码 — lib/Dialect/](https://github.com/triton-lang/triton/tree/main/lib/Dialect)
