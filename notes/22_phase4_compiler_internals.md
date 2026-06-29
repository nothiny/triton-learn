# 22 — Triton 编译器内部机制深度

> 深入 `@triton.jit` 的内部：AST 解析、lowering 全流程追踪、software pipelining、寄存器分配、autotuner 原理、PTX→SASS。
> 配合 `phase4_compiler/14-21` 教程系列。

---

## 0. 前言：你已经知道什么，这笔记新增什么

**已有笔记覆盖**（`21_triton_compiler_pipeline.md` + `20_triton_internals.md`）：
- 4 层 IR 管线（TTIR→TTGIR→LLVM→PTX）
- Layout encoding 系统的基础
- Pass 总览和关键 pass
- JIT 编译流程、cache 机制、环境变量

**这笔记新增**（Phase 4 进阶篇）：
- `@triton.jit` 内部——Python AST 如何变成 TTIR builder
- 一个 op 的完整 lowering 追踪——从 Python 到 PTX 每一步
- Software pipelining 的 cp.async 机制——比你想象的精巧
- 寄存器分配与 GPU SM 三资源约束——为什么 128 寄存器/线程是关键阈值
- Autotuner 的内部——不是什么"AI"，就是 grid search + cache
- PTX→SASS：最终机器码中的寄存器 spill 检测

---

## 1. `@triton.jit` 内部：Python AST → TTIR

### 1.1 JITFunction 对象

```python
@triton.jit
def my_kernel(x_ptr, N, BLOCK_SIZE: tl.constexpr):
    ...
```

`@triton.jit` 装饰器所做的：

1. **解析 Python AST**：用 `inspect.getsource(fn)` 拿源码，`ast.parse()` 生成 AST
2. **识别标记**：
   - `tl.constexpr` 注解 → 标记为编译期常量
   - `tl.load/tl.store/tl.dot/...` 调用 → 标记为 Triton DSL op
3. **构建 TTIR Builder**：遍历 AST，每个 `tl.*` 调用生成对应的 MLIR op 构建代码
4. **包装为 JITFunction**：`kernel[grid](args...)` 被拦截→触发编译或从 cache 加载

关键文件（Triton 源码中）：
- `python/triton/runtime/jit.py` — JITFunction 类
- `python/triton/language/semantic.py` — AST 语义分析器
- `python/triton/compiler/code_generator.py` — TTIR 生成器

### 1.2 tl.load 不是"执行"，而是"描述"

```python
# 在 @triton.jit 函数内部:
x = tl.load(x_ptr + offs, mask=mask)
```

这段代码在 JIT 编译时**不执行**。`tl.load()` 返回一个 `TensorDescriptor`——它记录"这里需要一次 load，参数是什么"，编译器稍后把它翻译成 `tt.load` MLIR op。

这就是为什么 Triton kernel 不能使用任意 Python：
- ❌ `while` — 编译期无法确定迭代次数
- ❌ `try/except` — 不是代数运算
- ❌ `len()` — 运行时函数，不是 DSL
- ✅ `for i in range(0, N, BLOCK)` — 步长是 constexpr，编译期可分析

### 1.3 tl.constexpr 的检测

```python
def kernel(x_ptr, N, BLOCK_SIZE: tl.constexpr):
    #                ^^^^^^^^^^^^^^^^^^^^
    # Python AST 中: ast.Attribute(value=Name('tl'), attr='constexpr')
```

Triton 的 AST visitor 遍历函数参数的 annotation：
- 找到 `tl.constexpr` → 标记为"编译期常量"
- 编译时：这些参数的值嵌入 IR（如 `arith.constant 256`）
- 运行时：从 kwargs 提取值，加入 cache key hash

不同 constexpr 值 → 不同的 cache key → 不同的编译产物。这就是 autotune 存在的基础。

> 🔧 **Compiler Perspective**：这类似于 partial evaluation——某些参数在编译期已知，编译器可以据此生成特化代码。传统 CPU 编译器用 `-D` 宏或 template 参数做类似的事。

---

## 2. 单操作 Lowering 全流程追踪：tl.dot 的旅程

这是理解 Triton 编译器最有效的方法：追踪一个 op 穿过每一层 IR。

### Stage 1: Python

```python
a = tl.load(A_ptr)      # (32, 16) fp16
b = tl.load(B_ptr)      # (16, 32) fp16
acc = tl.dot(a, b)      # (32, 32) fp32
```

### Stage 2: TTIR

```mlir
%a = tt.load %A_ptr : tensor<32x16xf16>
%b = tt.load %B_ptr : tensor<16x32xf16>
%acc = tt.dot %a, %b : tensor<32x16xf16> × tensor<16x32xf16> → tensor<32x32xf32>
```

关键：`tt.dot` 是一个**语义标记**——"这里需要矩阵乘法"。还没有决定用 Tensor Core 还是 CUDA Core。

### Stage 3: TTGIR (after ConvertTritonToTritonGPU)

```mlir
%a : tensor<32x16xf16, #dot_op<{opIdx=0, parent=#blocked<...>}>>
%b : tensor<16x32xf16, #dot_op<{opIdx=1, parent=#blocked<...>}>>
%acc : tensor<32x32xf32, #mma<{versionMajor=2, instrShape=[16,8,16]}>>
```

关键变化：
- 输入加了 `#dot_op` layout（MMA 操作数布局）
- 输出加了 `#mma` layout（Tensor Core 输出布局）
- Layout 转换可能被插入（`ttg.convert_layout`）

### Stage 3b: After AccelerateMatmul

```mlir
%acc = ttg.mma %a, %b {
    versionMajor=2, versionMinor=0,
    instrShape=[16,8,16]
}
```

`tt.dot` → `ttg.mma` 被替换。`instrShape=[16,8,16]` = Ampere 的 m16n8k16 MMA 指令。

### Stage 4: LLVM IR (after ConvertTritonGPUToLLVM)

```llvm
%mma = call @llvm.nvvm.mma.m16n8k16.row.col.f32.f16.f16.f32(...)
```

Layout 展开 → 线程索引计算（`threadIdx.x`, `blockIdx.x` 显式出现）。`ttg.mma` → NVVM MMA intrinsic。

### Stage 5: PTX

```asm
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    {%f1, %f2, %f3, %f4},    // D matrix fragments
    {%f5, %f6},               // A matrix fragments
    {%f7, %f8},               // B matrix fragments
    {%f1, %f2, %f3, %f4};    // C accumulator
```

一条 `mma.sync` = 16×8×16 = 4096 FLOPs/CUDA cycle。32 个线程协作完成一个 warp-level MMA。

### Stage 6: SASS (ptxas)

```asm
HMMA.16816.F32.F16.F16.F32 R0, R2, R4, R0;
```

物理寄存器编号（R0-R255）。可能包含 spill（`STL`/`LDL`）——PTX 中没有的但 SASS 中有的。

> 🔧 **Compiler Perspective**：这个 lowering 链展示了 MLIR 的"渐进 lowering"模式。每个 stage 只改变一个维度：TTIR→TTGIR 加 layout，AccelerateMatmul 替换 op，ConvertGPUToLLVM 展开 layout。这是 dialect-based compilation 的核心优势——每个 pass 职责单一，可以独立测试和调试。

---

## 3. Software Pipelining 的 cp.async 机制

### 3.1 串行 vs 流水线

```
num_stages=1 (串行):
  |-- load[0] --|-- compute[0] --|-- load[1] --|-- compute[1] --|

num_stages=2 (流水线):
  |-- load[0] --|-- load[1] --|-- load[2] --|
                  |-- compute[0] --|-- compute[1] --|
  (加载和计算在时间上重叠 → 总耗时减少 ~40%)
```

### 3.2 cp.async 的三指令模式

Triton 的 `TritonGPUPipeline` pass 把串行循环变成异步流水线：

```asm
// 1. 发起异步拷贝（线程不等待！）
cp.async.ca.shared.global [shared_buf_0], [global_ptr_A]
cp.async.ca.shared.global [shared_buf_0], [global_ptr_B]

// 2. 标记一个 commit group
cp.async.commit_group

// ... 其他计算 ...

// 3. 等待拷贝完成
cp.async.wait_group 0   // 0 = 等所有已提交的 group

// 4. 使用 shared memory 中的数据
ld.shared.f32 %f1, [shared_buf_0]
mma.sync ... %f1, ...
```

### 3.3 num_stages 和 shared memory 权衡

| num_stages | shared memory 用量 | 内存延迟隐藏 | 适用场景 |
|-----------|-------------------|------------|---------|
| 1 | 1 × tile | 无 | 小 kernel，不需要流水线 |
| 2 | 2 × tile（双缓冲） | 中等 | 大多数 GEMM |
| 3-4 | 3-4 × tile | 高 | 大 tile，内存密集 |
| 5+ | 5+ × tile | 很高 | 可能降低 occupancy |

`num_stages` 增加 → shared memory 增加 → 可能降低 occupancy → 需要在 autotune 中权衡。

---

## 4. 寄存器分配与三资源约束

### 4.1 GPU SM 的资源模型

```
每个 SM（以 H100 为例）:
  ┌──────────────────┬───────────┐
  │ 寄存器（32-bit）  │ 65536     │  ← 每线程最多 255
  │ Shared Memory    │ 228 KB    │
  │ 最大 Warps       │ 64        │
  │ 最大 Blocks      │ 32        │
  └──────────────────┴───────────┘
```

你的 kernel 每 CTA 需要：
- 寄存器 = `num_warps × 32 × registers_per_thread`
- Shared Memory = 从 `.shared` 声明中统计
- Warps = `num_warps`

**三者的最小值决定 occupancy**：一个 SM 能同时驻留多少个 CTA。

### 4.2 寄存器压力的"甜蜜点"

| 寄存器/线程 | 状态 | 行动 |
|-----------|------|------|
| < 64 | 很轻松 | 可以考虑增加 `num_warps` |
| 64-128 | 正常 | 保持 |
| 128-200 | 较高 | 注意 occupancy |
| > 200 | 很高 | 可能 spill，试试减小 BLOCK_SIZE |
| > 255 | H100 上限！ | 一定会 spill |

### 4.3 Triton 间接控制寄存器的方式

Triton 不做寄存器分配（LLVM 做）。但通过参数间接影响：

- `num_warps ↑` → 每 warp 可用寄存器 ↓ → 可能触发 spill
- `BLOCK_SIZE ↑` → `sizePerThread ↑` → 直接增加寄存器需求
- `num_stages ↑` → shared memory ↑ → 间接影响（挤占 occupancy）
- `dtype`：fp32 比 fp16 多用 1 倍寄存器宽度

### 4.4 Spill 检测

**PTX 级别**（需求）：
```bash
grep "st.local\|ld.local" ~/.triton/cache/*.ptx
# 如果有 → spill 一定会发生
```

**SASS 级别**（实际）：
```bash
cuobjdump -sass *.cubin | grep "STL\|LDL"
# STL = 溢出到 local memory
# LDL = 从 local memory 恢复
```

如果 PTX 没有 `st.local` 但 SASS 有 `STL` → ptxas 发现物理寄存器不够，自动 spill。这是看 SASS 的核心价值。

---

## 5. Autotuner 内部

### 5.1 本质：Grid Search，不是 AI

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 128}, num_warps=4),
        triton.Config({"BLOCK": 256}, num_warps=8),
        ...
    ],
    key=["N"],  # ← 影响性能的运行时参数
)
```

工作流程：
1. 首次调用 → 遍历所有 config
2. 每个 config：编译 kernel → GPU 预热（`warmup` 次）→ 测量耗时（`rep` 次）
3. 缓存最优 config 到 `~/.triton/cache/<hash>/autotune.json`
4. 后续调用 → 直接从 cache 读取最优 config

### 5.2 `key` 参数的设计

```python
# 好: key 是影响性能的形状参数
key=["M", "N", "K"]

# 坏: key 是 BLOCK_SIZE 或 num_warps
#     这些都是 config 参数，不是 cache key
key=["BLOCK_SIZE"]  # ❌ 没有意义
```

不同 `key` 值 → 不同 cache entry → 可能选不同 config。

### 5.3 Prune（剪枝）

```python
@triton.autotune(
    configs=[...],
    key=["N"],
    prune_configs_by={
        "early_config_prune": lambda configs, N: [
            c for c in configs
            if c.kwargs["BLOCK_SIZE"] <= N  # block 不能比数据大
        ]
    }
)
```

剪枝在编译前执行，避免浪费编译时间。

### 5.4 Cache 结构

```
~/.triton/cache/
├── <kernel_hash_A>/
│   ├── <config_1_hash>/
│   │   ├── *.ptx
│   │   └── *.cubin
│   ├── <config_2_hash>/
│   │   └── ...
│   └── __launcher_cache__/
│       └── autotune.json    ← {"<key_hash>": "<best_config_hash>"}
```

按 kernel hash + config hash 两级 cache。修改源码 → kernel hash 变 → 重新 autotune。

---

## 6. PTX → SASS

### 6.1 区别

| | PTX | SASS |
|---|-----|------|
| 性质 | 虚拟 ISA，跨架构兼容 | 物理机器码，架构特定 |
| 寄存器 | 虚拟寄存器（需求） | 物理寄存器（实际分配） |
| spill | 可能没有 `st.local` | 可能有 `STL`/`LDL` |
| 调度 | 顺序排列 | 指令重排 |
| 生成 | `llc -march=nvptx` | `ptxas` |

### 6.2 反汇编命令

```bash
# 从 cache 找到 cubin
CUBIN=$(ls -t ~/.triton/cache/*/*.cubin | head -1)

# cuobjdump (推荐)
cuobjdump -sass $CUBIN > kernel.sass

# 或 nvdisasm
nvdisasm -ndf -c $CUBIN > kernel.sass

# 分析
grep "STL\|LDL" kernel.sass    # spill 检测
grep "HMMA\|IMMA" kernel.sass  # Tensor Core
grep "R[0-9]" kernel.sass | sort -u | wc -l  # 物理寄存器数
```

### 6.3 关键：SASS 中的 Spill

PTX 显示需要 128 个虚拟寄存器 → ptxas 分配时发现物理寄存器只有 96 个可用 → 把 32 个值溢出到 local memory (stack) → SASS 中出现 `STL` (store local) 和 `LDL` (load local) → 每条 `STL`/`LDL` 增加 ~100-200 cycles → 性能大幅下降

PTX 中看不到 spill → 只有看 SASS 才能确认真正的寄存器分配结果

---

## 7. 参考文件

这些是 `phase4_compiler/` 中对应的可执行教程：

| 主题 | 教程文件 |
|------|---------|
| AST → TTIR | `14_ast_to_ttir.py` |
| 内存模型在各层 IR | `15_memory_model.py` |
| MMA 深度 | `16_mma_deep.py` |
| Lowering 全追踪 | `17_lowering_trace.py` |
| Autotuner | `18_autotuner.py` |
| 环境变量速查 | `19_env_vars.py` |
| 源码导航 | `20_source_guide.py` |
| PTX → SASS | `21_ptx_to_sass.py` |
| Pipelining 详解 | `09_pipeline_prefetch.py` |
| 寄存器压力 | `10_register_pressure.py` |
| IR 调试实战 | `11_debugging_with_ir.py` |

---

## 8. 总结

```
@triton.jit kernel
    │
    │ inspect.getsource() + ast.parse()
    ▼
Python AST ──→ semantic visitor ──→ TTIR Builder
                                        │
                                        ▼
                                    TTIR (tt dialect)
                                        │ ConvertTritonToTritonGPU
                                        ▼
                                    TTGIR (tt + ttg dialect)
                                        │ AccelerateMatmul → MMA
                                        │ Pipeline → cp.async
                                        │ Prefetch → 提前加载
                                        ▼
                                    LLVM IR (NVVM intrinsics)
                                        │ NVPTX backend
                                        ▼
                                    PTX (虚拟 ISA)
                                        │ ptxas
                                        ▼
                                    SASS (物理机器码)
```

每一步都在做两件事：**丢掉一些上层信息，加入一些下层信息**。理解这个过程，你就能在性能出问题时精确地定位到"编译器的哪一步没有做你期望的事"。

> 🔧 **为什么这很重要**：传统 GPU 编程中，你手动管理线程→数据的映射（CUDA kernel 中的 `threadIdx.x` 算术）。Triton 把这个工作交给了编译器——但如果你不理解编译器怎么做的，你就无法判断它做得对不对。
