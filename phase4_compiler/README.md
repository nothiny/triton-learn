# Phase 4 — Triton 编译器内部：从 Python 到 GPU 指令

> 理解 Triton 如何把你的 Python kernel 一步步变成 GPU 机器码。
> 基础篇 (01-13) 适合从未接触过编译器的读者。
> 进阶篇 (14-21) 适合想深入源码级理解的读者。

## 为什么你要学这个？

写 Triton kernel 时，你写的是 Python，但执行的是 GPU 指令。中间发生了什么？

```python
@triton.jit
def add(x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK) + tl.program_id(0) * BLOCK
    x = tl.load(x_ptr + offs, mask=offs < N)
    y = tl.load(y_ptr + offs, mask=offs < N)
    tl.store(out_ptr + offs, x + y, mask=offs < N)
```

这段代码会被编译成 4 层中间表示，最终变成 PTX 汇编：

```
Python 源码  →  TTIR  →  TTGIR  →  LLVM IR  →  PTX
  (你写的)     (纯数学)  (加线程分配) (通用底层) (GPU 指令)
```

理解这个过程中的每一步，你就能：
- **读懂** 编译器生成的代码，判断它是否做了你期望的优化
- **调试** 性能问题——为什么这个 kernel 慢？哪里多了不必要的 layout conversion？
- **优化** 你的 Triton 代码——知道什么样的写法会让编译器生成更好的指令

## 学习路径

### 基础篇 (01-13): IR 逐层解析 + Pass 管线 + 实战调试

按顺序阅读和运行。每个文件都是可执行的教程：

| # | 文件 | 内容 | 难度 |
|---|------|------|------|
| 01 | `01_first_ir.py` | 第一次接触 IR：dump 并阅读 4 层 IR | ⭐ |
| 02 | `02_ttir_language.py` | TTIR dialect 详解：每个 op 对应什么 Python 代码 | ⭐⭐ |
| 03 | `03_to_ttgir.py` | 关键一步：TTIR → TTGIR，layout encoding 出现了 | ⭐⭐ |
| 04 | `04_layout_system.py` | Layout 系统深度解析：5 种 layout 类型 | ⭐⭐⭐ |
| 05 | `05_convert_layout.py` | Layout 转换的代价——隐形的性能杀手 | ⭐⭐⭐ |
| 06 | `06_llvm_ir.py` | LLVM IR：寄存器、地址、分支 | ⭐⭐ |
| 07 | `07_ptx_assembly.py` | PTX 汇编精读：GPU 真正执行的指令 | ⭐⭐⭐ |
| 08 | `08_pass_pipeline.py` | Pass pipeline 全景：每个 pass 做什么 | ⭐⭐⭐ |
| 09 | `09_pipeline_prefetch.py` | Software pipelining：让加载和计算重叠 | ⭐⭐⭐⭐ |
| 10 | `10_register_pressure.py` | 寄存器分配与三资源约束 | ⭐⭐⭐ |
| 11 | `11_debugging_with_ir.py` | 实战：用 IR 诊断 4 种常见性能问题 | ⭐⭐⭐⭐ |
| 12 | `12_compile_api.py` | triton.compiler API：程序化访问编译管线 | ⭐⭐⭐ |
| 13 | `13_custom_pass.py` | 自定义 MLIR Pass 概念 + AST 分析工具 | ⭐⭐⭐⭐ |

### 进阶篇 (14-21): 源码级深入

需要基础篇全部完成，适合想深入了解底层机制的读者：

| # | 文件 | 内容 | 难度 |
|---|------|------|------|
| 14 | `14_ast_to_ttir.py` | @triton.jit 内部机制：AST 解析、JITFunction 解剖 | ⭐⭐⭐ |
| 15 | `15_memory_model.py` | 内存模型深度：HBM→Shared→Register 在各层 IR 的形态 | ⭐⭐⭐ |
| 16 | `16_mma_deep.py` | MMA/Tensor Core 深度：指令形状、Ampere vs Hopper | ⭐⭐⭐⭐ |
| 17 | `17_lowering_trace.py` | 单操作全流程追踪：tl.dot 从 Python 到 PTX 每一步 | ⭐⭐⭐⭐ |
| 18 | `18_autotuner.py` | Autotuner 内部机制：搜索算法、cache 结构、最佳实践 | ⭐⭐⭐ |
| 19 | `19_env_vars.py` | Triton 全部环境变量速查 + 调试工作流 | ⭐⭐ |
| 20 | `20_source_guide.py` | Triton 源码导航：Python/C++ 端关键文件阅读顺序 | ⭐⭐⭐ |
| 21 | `21_ptx_to_sass.py` | PTX→SASS：反汇编、寄存器分配验证、spilling 检测 | ⭐⭐⭐⭐ |

### MLIR 专题 (22-27): MLIR 框架深度

需要基础篇全部完成，适合想理解 Triton 编译器底层基础设施的读者：

| # | 文件 | 内容 | 难度 |
|---|------|------|------|
| 22 | `22_mlir_core_concepts.py` | MLIR 核心概念：Operation/Type/Attribute/Dialect | ⭐⭐⭐ |
| 23 | `23_mlir_text_format.py` | MLIR 文本格式语法详解 + 手工构造练习 | ⭐⭐⭐ |
| 24 | `24_triton_tt_dialect.py` | `tt` dialect 完整参考：每个 op 的语义/Python 映射 | ⭐⭐⭐ |
| 25 | `25_triton_ttg_dialect.py` | `ttg` dialect 完整参考：layout 类型/GPU op | ⭐⭐⭐⭐ |
| 26 | `26_mlir_pass_system.py` | MLIR Pass 基础设施 + Pattern Rewriting 详解 | ⭐⭐⭐⭐ |
| 27 | `27_ir_analysis_tools.py` | 构建 IR 分析工具箱：op 统计/配置对比/cache 分析 | ⭐⭐⭐ |

## 前置知识

- 会写 Triton kernel（Phase 1-3 水平）
- 知道 GPU 基本概念（thread, block, warp, shared memory, HBM）
- 基础篇不需要编译器背景；进阶篇建议有编译器基础

## 运行方式

```bash
# 基础篇
make run-compiler-01     # 01_first_ir.py
make run-compiler-02     # 02_ttir_language.py
# ... 以此类推到 13

# 进阶篇
make run-compiler-14     # 14_ast_to_ttir.py
# ... 以此类推到 21

# 或者直接
python phase4_compiler/01_first_ir.py
```

## 学完后的进阶

- 阅读 Triton 源码：`triton/lib/Conversion/`, `triton/lib/Dialect/`
- 用 `MLIR_PRINT_IR_AFTER_ALL=1` 观察每个 pass 前后的变化
- 对比 autotune 不同 config 生成的 PTX 差异
- 尝试写一个自定义 pass 来分析你的 kernel
- 用 `cuobjdump -sass` 分析 SASS，验证寄存器分配
