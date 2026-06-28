"""
22_mlir_core_concepts.py — MLIR 核心概念

学习目标:
  1. 理解 MLIR 是什么 (不是"一种 IR"，而是"构建 IR 的框架")
  2. 掌握 MLIR 的 5 个核心抽象: Operation, Type, Attribute, Dialect, Region
  3. 用 Triton 真实生成的 TTIR 来理解每个概念
  4. 理解 SSA (Static Single Assignment) 形式

运行: python phase4_compiler/22_mlir_core_concepts.py

前提: 已完成 Phase 4 基础篇 (01-13)。
"""

import os
import re
from pathlib import Path

os.environ["TRITON_KERNEL_DUMP"] = "1"
os.environ["TRITON_KERNEL_OVERRIDE"] = "1"

import torch
import triton
import triton.language as tl


# ══════════════════════════════════════════════════════════════════════
# 生成一些 IR 用于分析
# ══════════════════════════════════════════════════════════════════════


@triton.jit
def mlir_demo_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
    """
    一个简单的 kernel，足够展示 MLIR 的核心概念。
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    z = x + y
    tl.store(out_ptr + offs, z, mask=mask)


# ══════════════════════════════════════════════════════════════════════
# MLIR 文本解析器 (轻量级)
# ══════════════════════════════════════════════════════════════════════


class MLIRTextAnalyzer:
    """
    一个简单的 MLIR 文本分析器。
    不是完整的 MLIR parser，但足以提取结构和关键操作。
    用于教学目的——展示如何"读"MLIR。
    """

    def __init__(self, mlir_text: str):
        self.text = mlir_text

    def find_operations(self) -> list[dict]:
        """提取所有 MLIR operations 及其关键信息。"""
        ops = []
        # 匹配模式: %result = dialect.op_name {attrs} : types
        # 或: dialect.op_name {attrs} : types
        pattern = r'(\S+)\s*=\s*(\w+)\.(\w+)\s*(\{[^}]*\})?\s*:\s*([^\n;]+)'
        for m in re.finditer(pattern, self.text):
            ops.append({
                "result": m.group(1),
                "dialect": m.group(2),
                "op_name": m.group(3),
                "attrs": m.group(4) if m.group(4) else "",
                "types": m.group(5).strip(),
            })
        return ops

    def find_types(self) -> list[str]:
        """提取所有 MLIR 类型。"""
        # 找 !dialect.type<...> 和 tensor<...>
        types = set()
        # !tt.ptr<f32>
        for m in re.finditer(r'!\w+\.\w+(?:<[^>]*>)?', self.text):
            types.add(m.group())
        # tensor<256xf32>
        for m in re.finditer(r'tensor<\d+x[^>]+>', self.text):
            types.add(m.group())
        return sorted(types)

    def find_attributes(self) -> list[str]:
        """提取所有 MLIR attributes。"""
        # {key = value} 和 #dialect.attr<...>
        attrs = set()
        # tt.divisibility = 16 : i32
        for m in re.finditer(r'\{([^}]+)\}', self.text):
            # 解析 key = value 对
            inner = m.group(1)
            for pair in inner.split(","):
                pair = pair.strip()
                if "=" in pair:
                    attrs.add(pair)
        # #blocked<{...}>
        for m in re.finditer(r'#\w+(?:\.\w+)?<\{[^}]+\}>', self.text):
            attrs.add(m.group())
        return sorted(attrs)

    def extract_function(self) -> str | None:
        """提取函数的完整定义。"""
        m = re.search(r'(tt\.func\s+public\s+@\w+\s*\([^)]*\).*?\{.*?\n\s*\})',
                       self.text, re.DOTALL)
        return m.group(0) if m else None

    def count_dialects(self) -> dict[str, int]:
        """统计各 dialect 的 op 出现次数。"""
        counts = {}
        for m in re.finditer(r'(\w+)\.(\w+)', self.text):
            dialect = m.group(1)
            if dialect not in ("reg", "param", "global"):  # skip non-dialect
                counts[dialect] = counts.get(dialect, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))


# ══════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════


def main():
    print("=" * 70)
    print("  22 — MLIR 核心概念")
    print("=" * 70)

    # ═══════════════════════════════════════════════════════════
    # 1. MLIR 是什么?
    # ═══════════════════════════════════════════════════════════
    print("─" * 70)
    print("  1. MLIR 不是一种 IR，而是构建 IR 的框架")
    print("─" * 70)
    print("""
  ╔══════════════════════════════════════════════════════════════════╗
  ║  MLIR = Multi-Level Intermediate Representation                  ║
  ║  "Multi-Level" 不是指"多层抽象"，而是指"支持多种 IR 共存"          ║
  ╚══════════════════════════════════════════════════════════════════╝

  传统编译器:
    AST → 一种统一的 IR → 机器码
    (所有语言、所有优化都在同一个 IR 上做)

  MLIR 编译器框架:
    Python AST → TTIR → TTGIR → LLVM IR → PTX
    (每个阶段是不同 dialect 的 IR，MLIR 提供基础设施让它们"对话")

  MLIR 提供:
    • Operation (操作):            IR 的基本单位，类似指令
    • Type (类型):                 值的类型系统
    • Attribute (属性):           编译期常量元数据
    • Dialect (方言):             特定领域的 op/type/attr 集合
    • Region & Block (区域和块):  控制流结构
    • Pass Framework:             对 IR 做变换的统一接口

  为什么要了解 MLIR?
    • Triton 的 IR 就是 MLIR — 理解 MLIR 就是理解 Triton 的编译过程
    • 读懂 .ttir/.ttgir 文件 = 理解编译器在"想什么"
    • 写自定义 pass 需要理解 MLIR 的 pass 机制
""")

    # ═══════════════════════════════════════════════════════════
    # 2. 生成并加载真实的 TTIR
    # ═══════════════════════════════════════════════════════════
    print("─" * 70)
    print("  2. 加载真实的 Triton TTIR")
    print("─" * 70)

    N = 256
    x = torch.randn(N, device="cuda")
    y = torch.randn(N, device="cuda")
    out = torch.empty(N, device="cuda")
    mlir_demo_kernel[(1,)](x, y, out, N, BLOCK=256)
    torch.cuda.synchronize()

    cache = Path.home() / ".triton" / "cache"
    ttir_files = sorted(cache.rglob("*.ttir"),
                         key=lambda p: p.stat().st_mtime, reverse=True)
    if not ttir_files:
        print("  ⚠ 未找到 TTIR 文件")
        return

    ttir_path = ttir_files[0]
    ttir_text = ttir_path.read_text()
    print(f"  文件: {ttir_path.name}")
    print(f"  大小: {len(ttir_text)} 字符")

    # ═══════════════════════════════════════════════════════════
    # 3. 核心概念 #1: Operation (操作)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  3. 核心概念 1/5: Operation (操作)")
    print("─" * 70)

    analyzer = MLIRTextAnalyzer(ttir_text)
    ops = analyzer.find_operations()

    print(f"""
  Operation = MLIR 的基本计算单元。
  类似 LLVM IR 的 instruction 或 PTX 的一条指令。

  格式:
    %result = dialect.op_name {{attributes}} : (input_types) -> output_type

  在真实 TTIR 中提取到的前 10 个 Operation:""")

    for i, op in enumerate(ops[:10]):
        print(f"\n    [{i+1}] {op['dialect']}.{op['op_name']}")
        print(f"        结果: {op['result']}")
        print(f"        返回类型: {op['types']}")
        if op['attrs']:
            print(f"        属性: {op['attrs']}")

    print(f"""
  🔑 Operation 的解剖:
    %pid = tt.get_program_id x : i32
    │      │   │               │  │
    │      │   │               │  └─── 返回类型 (output type)
    │      │   │               └─────── 操作数 (operands: 'x')
    │      │   └──────────────────────── 操作名 (op name)
    │      └───────────────────────────── dialect (tt = Triton)
    └───────────────────────────────────── 结果值 (SSA value)

  关键:
    • 每个 op 有 0 或多个 operands (输入值)
    • 每个 op 有 0 或多个 results (输出值)
    • op name 是 "dialect.op" 格式 (如 tt.load, arith.addf)
    • types 紧跟在 ':' 后面
""")

    # ═══════════════════════════════════════════════════════════
    # 4. 核心概念 #2: Type (类型)
    # ═══════════════════════════════════════════════════════════
    print("─" * 70)
    print("  4. 核心概念 2/5: Type (类型)")
    print("─" * 70)

    types = analyzer.find_types()
    print(f"\n  在 TTIR 中发现的类型:")
    for t in types:
        print(f"    {t}")

    print(f"""
  MLIR 类型系统:
    • 每个 SSA value 有确定的类型
    • 类型由 dialect 定义 — 不同 dialect 可以定义自己的类型
    • 类型在 op 的签名中声明 (在 ':' 后面)

  Triton 使用的主要类型:

  ┌──────────────────────┬─────────────────────────────────────────┐
  │ 类型                  │ 含义                                    │
  ├──────────────────────┼─────────────────────────────────────────┤
  │ i32, f32, f16        │ 标量类型 (来自 builtin dialect)         │
  │ !tt.ptr<f32>         │ Triton 指针 (指向 f32 的指针)          │
  │ tensor<256xf32>       │ 256 个 f32 元素的一维 tensor           │
  │ tensor<32x64xf16>     │ 32×64 的 f16 二维 tensor              │
  │ tensor<32x64xf32,     │ 带 layout encoding 的 tensor (TTGIR)  │
  │   #blocked<{...}>>   │                                        │
  └──────────────────────┴─────────────────────────────────────────┘

  关键区别:
    • !tt.ptr<f32>:   一个指针值 (标量)，不是 tensor
    • tensor<256xf32>: 256 个 f32 元素的 tensor (有 shape)
    • tensor<..., #blocked<...>>: 只在 TTGIR 中出现，带了线程分布信息
""")

    # ═══════════════════════════════════════════════════════════
    # 5. 核心概念 #3: Attribute (属性)
    # ═══════════════════════════════════════════════════════════
    print("─" * 70)
    print("  5. 核心概念 3/5: Attribute (属性)")
    print("─" * 70)

    attrs = analyzer.find_attributes()
    print(f"\n  在 TTIR 中发现的 attributes:")
    for a in attrs:
        print(f"    {a}")

    print(f"""
  Attribute = 编译期常量元数据，附加在 op 或 type 上。

  与 Operand 的区别:
    • Operand: 运行时值 (由其他 op 产生，通过 SSA 传递)
    • Attribute: 编译期常量 (编译时就知道了，不会变化)

  例子:
    %c = arith.constant 256 : i32
    #                      ↑
    #                      256 是 arith.constant 的属性 (编译期已知)

    tt.load %ptr, %mask : ...
    #        ↑     ↑
    #        operands (运行时值)

    tt.divisibility = 16 : i32
    #  ↑
    #  这是 attribute，告诉编译器"这个指针 16 字节对齐"

  Triton 的关键 attributes:
    • tt.divisibility:       指针对齐信息 (用于优化 load/store)
    • #ttg.blocked<{...}>:   layout encoding (TTGIR 中最重要的 attribute!)
    • #ttg.mma<{...}>:       Tensor Core MMA layout
    • ttg.num-warps, ttg.num-ctas:  module 级别的编译参数
""")

    # ═══════════════════════════════════════════════════════════
    # 6. 核心概念 #4: Dialect (方言)
    # ═══════════════════════════════════════════════════════════
    print("─" * 70)
    print("  6. 核心概念 4/5: Dialect (方言)")
    print("─" * 70)

    dialect_counts = analyzer.count_dialects()
    print(f"\n  TTIR 中的 dialect 分布:")
    for dialect, count in dialect_counts.items():
        print(f"    {dialect}: {count} 个 op")

    print(f"""
  Dialect = 一组相关的 op、type、attribute 的命名空间。

  Triton 编译器涉及的关键 dialect:

  ┌─────────────┬──────────────────────────────────────────────────┐
  │ Dialect     │ 说明                                              │
  ├─────────────┼──────────────────────────────────────────────────┤
  │ builtin     │ MLIR 内置: module, func (没有前缀的 op)           │
  │ arith       │ 算术运算: addf, mulf, addi, cmpi, constant       │
  │ tt          │ ★ Triton IR: load, store, dot, reduce, ...      │
  │ ttg         │ ★ Triton GPU IR: convert_layout, async_copy     │
  │ scf         │ 结构化控制流: for, if, while                     │
  │ cf          │ 低级控制流: br, cond_br                           │
  │ math        │ 数学函数: sqrt, exp, sin, ...                     │
  │ nvvm        │ NVIDIA 特定 (LLVM IR 阶段)                       │
  └─────────────┴──────────────────────────────────────────────────┘

  为什么需要 Dialect?
    • 隔离: arith 不需要知道 GPU，tt 不需要知道控制流
    • 渐进 lowering: 高层 dialect op → 低层 dialect op
    • 可扩展: 任何人都可以定义自己的 dialect

  Triton 的两大自定义 dialect:
    tt  (Triton):   "做什么运算" — 纯数学描述，无 GPU 信息
    ttg (TritonGPU): "怎么做" — 带 layout encoding，有 GPU 线程分配
""")

    # ═══════════════════════════════════════════════════════════
    # 7. 核心概念 #5: Region, Block, SSA
    # ═══════════════════════════════════════════════════════════
    print("─" * 70)
    print("  7. 核心概念 5/5: Region, Block, 和 SSA")
    print("─" * 70)

    # 找函数体
    func = analyzer.extract_function()
    if func:
        print(f"\n  函数定义 (前 500 字符):")
        for line in func[:500].split("\n"):
            print(f"    {line}")

    print(f"""
  MLIR 的结构层次:

  Module                          ← 最外层容器
  ├── Function (tt.func @name)   ← 函数 = 一个 kernel 入口
  │   └── Region                  ← 函数体是一个 Region
  │       └── Block               ← Region 包含 Block(s)
  │           ├── %pid = tt.get_program_id x : i32    ← Operation
  │           ├── %c = arith.constant 256 : i32        ← Operation
  │           ├── %offsets = tt.make_range {...}      ← Operation
  │           └── tt.return                            ← Terminator

  Region = 一组 Block 的容器 (通常只有一个 Block = 函数体)
  Block  = 一组 Operation 的列表，最后一个 op 必须是 Terminator
  Terminator = 特殊的 op，标记 Block 结束 (如 tt.return, scf.yield)

  SSA (Static Single Assignment):
    • 每个值只能被定义一次 (single assignment)
    • 每个值在使用前必须被定义 (static)
    • 值以 %name 引用

  例 (TTIR 中的 SSA):
    %pid = tt.get_program_id x : i32          ← 定义 %pid
    %block_start = arith.muli %pid, %c256    ← 使用 %pid, 定义 %block_start
    %offsets = tt.make_range {...}           ← 定义 %offsets
    %offsets_1 = arith.addi %offsets, ...    ← 使用 %offsets, 定义 %offsets_1

    %pid 只定义了一次 → 符合 SSA
    每次使用都通过 %name 引用 → 清晰的 use-def chain

  为什么 SSA?
    • 编译器优化 (CSE, DCE, copy propagation) 更容易
    • use-def chain 明确 → 数据流分析更简单
    • 寄存器分配 (graph coloring) 可以直接用 SSA 图
""")

    # ═══════════════════════════════════════════════════════════
    # 8. 对比 TTIR 和 TTGIR
    # ═══════════════════════════════════════════════════════════
    print("─" * 70)
    print("  8. TTIR vs TTGIR: dialect 的渐进 lowering")
    print("─" * 70)

    ttgir_files = sorted(cache.rglob("*.ttgir"),
                          key=lambda p: p.stat().st_mtime, reverse=True)
    if ttgir_files:
        ttgir_analyzer = MLIRTextAnalyzer(ttgir_files[0].read_text())
        ttgir_dialects = ttgir_analyzer.count_dialects()
        print(f"\n  TTIR dialect 分布:  {dict(dialect_counts)}")
        print(f"  TTGIR dialect 分布: {dict(ttgir_dialects)}")

        ttgir_types = ttgir_analyzer.find_types()
        ttgir_blocked = [t for t in ttgir_types if "blocked" in t.lower()]
        print(f"\n  TTGIR 中新增的类型 (带 layout):")
        for t in ttgir_blocked:
            print(f"    {t}")

    print(f"""
  🔑 TTIR → TTGIR: "同一个 op 出现了两次? 不是的 — dialect 变了!"

  TTIR:
    %x = tt.load %ptr, %mask : tensor<256xf32>
    #     ↑↑ tt dialect

  TTGIR:
    %x = tt.load %ptr, %mask : tensor<256xf32, #blocked<{...}>>
    #     ↑↑ 还是 tt.load, 但类型多了 layout!
    #     注意: 这里 op 还是 tt.load, 并没有变成 ttg.load

  关键洞察:
    • 大多数 op 没有从 tt 变成 ttg — load 还是 tt.load
    • 变化的是 TYPE (加了 layout encoding attribute)
    • 只有少数 op 是 ttg 独有的: ttg.convert_layout, ttg.async_copy

  这体现了 MLIR 的设计哲学:
    • Dialect A 的 op 可以使用 Dialect B 的类型
    • Type 和 Attribute 可以跨 dialect 使用
    • 不需要为每个 lowering 阶段重新定义所有 op
""")

    # ═══════════════════════════════════════════════════════════
    # 9. MLIR 概念速查表
    # ═══════════════════════════════════════════════════════════
    print("─" * 70)
    print("  9. MLIR 核心概念速查")
    print("─" * 70)
    print("""
  ┌─────────────────┬──────────────────────────────────────────────────┐
  │ 概念              │ 在 Triton 中的体现                               │
  ├─────────────────┼──────────────────────────────────────────────────┤
  │ Operation        │ tt.load, tt.store, arith.addf, tt.dot           │
  │ Type             │ tensor<256xf32>, !tt.ptr<f32>, i32              │
  │ Attribute        │ #blocked<{...}>, tt.divisibility, num-warps     │
  │ Dialect          │ tt (Triton IR), ttg (Triton GPU), arith, scf   │
  │ Region           │ 函数体 (tt.func 的 body)                         │
  │ Block            │ 函数体中的 op 序列                                │
  │ SSA Value        │ %pid, %offsets, %x, %y                          │
  │ Module           │ 整个 .ttir 文件的顶层容器                         │
  │ Pass             │ ConvertTritonToTritonGPU, AccelerateMatmul, ... │
  │ Terminator       │ tt.return, scf.yield                             │
  └─────────────────┴──────────────────────────────────────────────────┘

  MLIR 文本格式速读:
    %result = dialect.op_name operands {attrs} : (inputs) -> outputs
    │        │         │     │         │         │
    │        │         │     │         │         └─── 输出类型
    │        │         │     │         └─────────── 属性 (编译期常量)
    │        │         │     └─────────────────────── 操作数 (运行时值)
    │        │         └─────────────────────────────── 操作名
    │        └───────────────────────────────────────── dialect
    └────────────────────────────────────────────────── SSA 值名
""")

    print("\n📖 下一步: python phase4_compiler/23_mlir_text_format.py")
    print("   深入学习 MLIR 文本格式的语法，学会精确读写 MLIR。\n")


if __name__ == "__main__":
    main()
