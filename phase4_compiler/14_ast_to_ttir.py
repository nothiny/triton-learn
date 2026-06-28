"""
14_ast_to_ttir.py — Python AST → TTIR: @triton.jit 的内部机制

学习目标:
  1. 理解 @triton.jit 如何解析你的 Python 代码
  2. 看到 tl.load / tl.store 等"调用"实际上是在构建 AST 节点
  3. 理解 tl.constexpr 的检测机制
  4. 知道为什么有些 Python 语法在 kernel 中不能用

运行: python phase4_compiler/14_ast_to_ttir.py

前提: 已完成 01-13。
"""

import ast
import inspect
import sys

import torch
import triton
import triton.language as tl


# ══════════════════════════════════════════════════════════════════════
# Part 1: @triton.jit 包装了什么？
# ══════════════════════════════════════════════════════════════════════


def inspect_jit_function():
    """深入检查 JITFunction 对象的内部结构。"""
    print("─" * 70)
    print("  1. JITFunction 对象解剖")
    print("─" * 70)

    @triton.jit
    def sample_kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(x_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x * 2.0, mask=mask)

    # JITFunction 的关键属性
    print(f"\n  type(sample_kernel) = {type(sample_kernel)}")
    print(f"  sample_kernel.__name__ = {sample_kernel.__name__}")

    # fn: 原始的 Python 函数
    print(f"\n  .fn 属性: {sample_kernel.fn}")
    print(f"  .fn 的类型: {type(sample_kernel.fn)}")
    print(f"  .fn 的源码:")
    src = inspect.getsource(sample_kernel.fn)
    for line in src.strip().split("\n")[:10]:
        print(f"    {line}")

    # arg_names / arg_key: 参数信息
    if hasattr(sample_kernel, 'arg_names'):
        print(f"\n  .arg_names: {sample_kernel.arg_names}")
    if hasattr(sample_kernel, 'constexprs'):
        print(f"  .constexprs: {sample_kernel.constexprs}")
    if hasattr(sample_kernel, 'do_not_specialize'):
        print(f"  .do_not_specialize: {sample_kernel.do_not_specialize}")

    # params: @triton.jit 接收的参数
    print(f"\n  JITFunction 的关键属性列表:")
    interesting = ['fn', 'arg_names', 'constexprs', 'do_not_specialize',
                    'cache_key', 'signature', 'num_warps', 'num_stages']
    for attr in interesting:
        if hasattr(sample_kernel, attr):
            val = getattr(sample_kernel, attr)
            if callable(val):
                print(f"    .{attr}: <callable>")
            else:
                print(f"    .{attr}: {val}")
        else:
            print(f"    .{attr}: (not present)")

    return sample_kernel


# ══════════════════════════════════════════════════════════════════════
# Part 2: tl.* 调用实际上在构建什么？
# ══════════════════════════════════════════════════════════════════════


def trace_triton_language_ops():
    """
    关键洞察: 在 @triton.jit 函数中调用 tl.load() 时，
    这个调用不是在"执行"加载操作，而是在构建一个 AST 节点。

    这类似于 PyTorch 的 torch.no_grad() 中的操作构建计算图。
    但 Triton 更进一步 — 它是在 JIT 编译时解析你的函数源码。
    """
    print("\n" + "─" * 70)
    print("  2. tl.load / tl.store 不是「执行」，而是「描述」")
    print("─" * 70)

    print("""
  在 @triton.jit 函数内部，tl.load() 等调用并不真正执行。
  相反，它们构建一个"语义描述"的 AST，Triton 编译器稍后会分析它。

  执行流程:
    kernel[grid](args...)
      → JITFunction.__call__
        → 检查 cache (有没有这个 kernel+参数的编译产物?)
        → (如果没有) 触发编译:
            1. inspect.getsource(kernel.fn) → 拿 Python 源码
            2. ast.parse(source) → 生成 Python AST
            3. Triton 的 AST visitor (triton/language/semantic.py)
               遍历 Python AST，识别 tl.* 调用
            4. 为每个 tl.* 调用生成对应的 MLIR op
            5. 构建完整的 TTIR module
            → 运行 compiler passes → 最终生成 CUBIN

  这意味着:
    • triton.language 是一个 DSL (Domain Specific Language)
    • 你在 kernel 中写的不是"普通 Python"，而是受约束的 DSL
    • 编译器只能处理它能识别的模式""")

    # 展示: tl.arange 返回的不是 tensor，而是一个 "placeholder"
    # 在编译期，这个 placeholder 会被替换成真正的 MLIR value
    print("""
  具体来说，每个 tl.* 函数返回什么?

    tl.arange(0, BLOCK)    → 返回一个 TensorDescriptor (不是真实数据!)
                            → 编译后变成 tt.make_range {start=0, end=BLOCK}
    tl.load(ptr, mask=...) → 返回一个 TensorDescriptor
                            → 编译后变成 tt.load
    tl.dot(a, b)           → 返回一个 TensorDescriptor
                            → 编译后变成 tt.dot (→ 然后被 AccelerateMatmul 替换)

  这些 TensorDescriptor 在编译期被收集、分析，然后生成 MLIR。
  在运行期它们不存在 — 只有生成的 PTX/CUBIN 在 GPU 上执行。
""")


# ══════════════════════════════════════════════════════════════════════
# Part 3: tl.constexpr 是如何工作的？
# ══════════════════════════════════════════════════════════════════════


def explain_constexpr():
    """深入解释 tl.constexpr 的机制。"""
    print("─" * 70)
    print("  3. tl.constexpr 的检测机制")
    print("─" * 70)

    print("""
  tl.constexpr 是 Triton JIT 编译的核心机制之一。

  检测过程:
    1. Triton 解析 kernel 函数的 Python AST
    2. 遍历函数参数
    3. 检查每个参数的 type annotation 是否是 tl.constexpr
    4. 如果是 → 标记为 "必须在编译期已知"
    5. 运行时:
       - 从 kwargs 中提取 constexpr 参数的值
       - 将它们加入 cache key hash
       - 编译时把它们作为编译期常量嵌入 IR

  示例:
    def kernel(x_ptr, N, BLOCK_SIZE: tl.constexpr):
        #                ^^^^^^^^^^^^^^^^^^^^
        #                Python AST 中这是 ast.Attribute(value=Name('tl'), attr='constexpr')
        #                Triton 的 visitor 识别这个 pattern

  为什么需要 constexpr?
    • GPU kernel 的 grid/block 大小必须在编译期确定
    • 循环展开、shared memory 大小等也是如此
    • constexpr 参数的不同值 → 不同的编译结果 → 不同的 PTX

  constexpr 参数 和 普通参数 的本质区别:
    • 普通参数 (N): 编译时不嵌入 IR，作为 kernel 的运行时参数传递
      → 可以在运行时变化，不触发重编译
    • constexpr 参数 (BLOCK_SIZE): 编译时嵌入 IR
      → 变化时触发重编译，产生不同的机器码

  在 TTIR 中的表现:
    • N (普通):    保留为函数参数  %n = tt.func_arg
    • BLOCK_SIZE:   直接替换为常数   %c256_i32 (一个立即数常量)
""")


# ══════════════════════════════════════════════════════════════════════
# Part 4: 为什么有些 Python 语法不能用？
# ══════════════════════════════════════════════════════════════════════


def explain_ast_limitations():
    """解释 Triton 的 AST 分析限制。"""
    print("─" * 70)
    print("  4. Triton AST 分析的限制")
    print("─" * 70)

    print("""
  Triton 的 Python AST 分析器是一个受限的 Python 子集编译器。
  它只能处理它能识别的 AST 节点。

  ✅ 支持的 Python AST 节点:
    • ast.FunctionDef    — 函数定义
    • ast.For             — for 循环 (range-based)
    • ast.If / ast.IfExp  — 条件语句/表达式
    • ast.BinOp           — 二元运算 (+, -, *, /, ...)
    • ast.UnaryOp         — 一元运算 (-, not, ...)
    • ast.Compare         — 比较 (<, >, ==, ...)
    • ast.Call             — 函数调用 (只限于 tl.* 和白名单函数)
    • ast.Subscript       — 下标访问 ([], [:, None], ...)
    • ast.Attribute       — 属性访问 (tl.load, ...)
    • ast.Tuple / ast.List — 元组和列表字面量
    • ast.Constant        — 常量 (1, 2.0, True, ...)
    • ast.Name / ast.Assign — 变量定义和引用

  ❌ 不支持的:
    • ast.While           — while 循环 (编译期无法确定迭代次数)
    • ast.Try / ast.Raise — 异常处理
    • ast.With             — with 语句
    • ast.Lambda           — lambda 表达式
    • ast.ListComp         — 列表推导式
    • ast.DictComp         — 字典推导式
    • ast.GeneratorExp     — 生成器表达式
    • ast.Yield / ast.YieldFrom — 生成器
    • ast.ClassDef         — 类定义
    • ast.Import / ast.ImportFrom — import 语句 (kernel 内部)
    • 大多数 Python 标准库函数调用

  为什么有限制?
    Triton 的编译器在做"符号执行"——它需要静态地知道:
      • 循环多少次?
      • 每个变量是什么类型?
      • 数据的 shape 是什么?
    动态特性 (while, try/except, generator) 无法在编译期静态分析。

  常见错误:
    >>> for i in range(N):         # ❌ N 是运行时参数!
    >>>     x = tl.load(ptr + i)

    正确:
    >>> for i in range(0, N, BLOCK):  # ✅ 循环步长是 constexpr
    >>>     x = tl.load(ptr + i)
""")

    # 演示: 用 Python AST 验证
    print("\n  ▸ 验证: 用 ast 模块解析一个 kernel 函数")
    import triton.language as tl

    @triton.jit
    def demo_constraints(a_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        # for i in range(N):  ← 如果取消注释，编译会失败
        for i in range(0, N, BLOCK):  # ← 正确: 步长是 constexpr
            x = tl.load(a_ptr + offs + i, mask=(offs + i) < N)
        return x

    src = inspect.getsource(demo_constraints.fn)
    tree = ast.parse(__import__('textwrap').dedent(src))
    # 统计 AST 节点类型
    node_types = {}
    for node in ast.walk(tree):
        t = type(node).__name__
        node_types[t] = node_types.get(t, 0) + 1
    print(f"    AST 节点类型分布:")
    for t, count in sorted(node_types.items(), key=lambda x: -x[1])[:10]:
        print(f"      {t}: {count}")


# ══════════════════════════════════════════════════════════════════════
# Part 5: Triton 源码中的关键文件
# ══════════════════════════════════════════════════════════════════════


def source_code_guide():
    """指引阅读 Triton 源码。"""
    print("\n" + "─" * 70)
    print("  5. Triton 源码导航: AST → TTIR 相关文件")
    print("─" * 70)

    # 找 triton 安装位置
    import triton
    triton_path = triton.__path__[0]
    print(f"\n  Triton 安装路径: {triton_path}")

    # 列出关键目录
    import os
    key_dirs = [
        "language/",           # tl.load, tl.store, tl.dot 等的定义
        "compiler/",           # 编译器入口
        "runtime/",            # JIT 编译运行时
    ]
    for d in key_dirs:
        full = os.path.join(triton_path, d)
        if os.path.isdir(full):
            files = [f for f in os.listdir(full) if f.endswith('.py')]
            print(f"    {d}: {files}")

    print(f"""
  🔑 AST → TTIR 的关键源码路径:

  python/triton/
    language/
      __init__.py        ← tl.load, tl.store, tl.dot 等 API 定义
      semantic.py        ← AST 语义分析: 把 Python AST 转为 Triton IR builder
      core.py            ← 核心类型: tensor, pointer, block 等

    compiler/
      __init__.py        ← compile() 入口 (如果暴露)
      code_generator.py  ← 从 AST 生成 TTIR MLIR

    runtime/
      jit.py             ← JITFunction 类: @triton.jit 的实现
      driver.py          ← CUDA driver 交互
      cache.py           ← JIT 缓存管理

  lib/  (C++ 源码, 在 GitHub 上查看):
    Conversion/
      TritonToTritonGPU/     ← ConvertTritonToTritonGPU pass
      TritonGPUToLLVM/       ← ConvertTritonGPUToLLVM pass
    Dialect/
      Triton/                ← tt dialect 定义
      TritonGPU/             ← ttg dialect 定义 + layout encoding

  建议阅读顺序:
    1. python/triton/runtime/jit.py        ← 理解 @triton.jit 的入口
    2. python/triton/language/semantic.py  ← 理解 AST → IR builder
    3. lib/Conversion/TritonToTritonGPU/   ← 理解最关键的 pass
""")


# ══════════════════════════════════════════════════════════════════════
# Part 6: 手动查看 JIT 编译的中间步骤
# ══════════════════════════════════════════════════════════════════════


def demo_jit_steps():
    """演示 JIT 编译的可观察步骤。"""
    print("─" * 70)
    print("  6. JIT 编译的可观察步骤")
    print("─" * 70)

    import os
    os.environ["TRITON_KERNEL_DUMP"] = "1"
    os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

    @triton.jit
    def tiny_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(x_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x * 2.0, mask=mask)

    N = 128
    x = torch.randn(N, device="cuda")
    out = torch.empty(N, device="cuda")

    print("  运行 kernel...")
    tiny_kernel[(1,)](x, out, N, BLOCK=128)
    torch.cuda.synchronize()
    print("  ✅ 编译并执行成功")

    # 查看 cache
    from pathlib import Path
    cache = Path.home() / ".triton" / "cache"
    if cache.exists():
        recent = sorted(cache.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)[:3]
        print(f"\n  最近的 cache 目录 (前 3 个):")
        for d in recent:
            if d.is_dir():
                contents = list(d.iterdir())[:5]
                print(f"    {d.name}/")
                for f in contents:
                    print(f"      {f.name}")

    print("""
  JIT 编译的时间线:
    1. kernel[grid](args...)                     ← 你的调用
    2. JITFunction.__call__                       ← 进入 JIT
    3. hash(src, constexprs, arch, version)        ← 计算 cache key
    4. cache lookup                                ← 找 ~/.triton/cache/<key>/
    5. (miss) inspect.getsource(fn)               ← 拿源码
    6. (miss) ast.parse(source)                   ← 解析 AST
    7. (miss) semantic visitor                     ← AST → IR builder
    8. (miss) MLIR passes (TTIR→TTGIR→LLVM→PTX)  ← 编译管线
    9. (miss) cuModuleLoad(cubin)                 ← 加载到 driver
    10. cuLaunchKernel(...)                       ← 启动!
""")


# ══════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════


def main():
    print("=" * 70)
    print("  14 — Python AST → TTIR: @triton.jit 内部机制")
    print("=" * 70)

    inspect_jit_function()
    trace_triton_language_ops()
    explain_constexpr()
    explain_ast_limitations()
    source_code_guide()
    demo_jit_steps()

    print("\n📖 下一步: python phase4_compiler/15_memory_model.py")
    print("   深入 Triton 的内存模型: HBM → Shared → Register 在各层 IR 的表现。\n")


if __name__ == "__main__":
    main()
