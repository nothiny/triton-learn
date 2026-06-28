"""
23_mlir_text_format.py — MLIR 文本格式语法详解

学习目标:
  1. 精通 MLIR 文本格式的每个语法元素
  2. 能手工解析和构造 MLIR 文本
  3. 理解 SSA 值编号、use-def chain、区域嵌套

运行: python phase4_compiler/23_mlir_text_format.py

前提: 已完成 22。
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
# 生成 IR
# ══════════════════════════════════════════════════════════════════════


@triton.jit
def format_demo(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + 1.0, mask=mask)


def get_fresh_ttir():
    cache = Path.home() / ".triton" / "cache"
    files = sorted(cache.rglob("*.ttir"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0].read_text() if files else None


# ══════════════════════════════════════════════════════════════════════
# MLIR 文本格式解析器
# ══════════════════════════════════════════════════════════════════════


class MLIRSyntaxExplainer:
    """逐元素解释 MLIR 文本格式的语法。"""

    @staticmethod
    def explain_module_header(mlir_text):
        """解释 module 头部。"""
        # 找 module 声明
        m = re.search(r'module\s*({[^}]*})?\s*\{', mlir_text)
        if m:
            header = m.group(0)
            return f"""
  module 声明:
    格式: module [{{attrs}}] {{
    例子: {header.strip()}

    解读:
      module 是 MLIR 的最外层容器，包含所有函数和全局定义。
      module attributes (可选) 指定全局属性，如 GPU target。
  """
        return "  (未找到 module 声明)"

    @staticmethod
    def explain_function_signature(mlir_text):
        """解释函数签名。"""
        m = re.search(
            r'(tt\.func)\s+(@\w+)\s*\(([^)]*)\)\s*([^{]*)',
            mlir_text
        )
        if m:
            return f"""
  函数签名:
    格式: dialect.func [visibility] @name (args...) [attributes] [-> return_types]
    例子:
      tt.func public @format_demo(
        %x_ptr: !tt.ptr<f32> {{tt.divisibility = 16 : i32}},
        %out_ptr: !tt.ptr<f32> {{tt.divisibility = 16 : i32}},
        %N: i32 {{tt.divisibility = 16 : i32}}
      )

    解读:
      tt.func     ← dialect 是 tt, 操作是 func
      public      ← 可见性 (GPU kernel 必须是 public)
      @format_demo ← 函数符号名 (@前缀)
      %x_ptr: !tt.ptr<f32> ← 参数: SSA名: 参数类型
      {{tt.divisibility = 16 : i32}} ← 参数 attribute (对齐信息)
  """
        return "  (未找到函数签名)"

    @staticmethod
    def explain_ssa_naming(mlir_text):
        """解释 SSA 值的命名规则。"""
        # 找 %name 和 %digit 两种格式
        named = len(re.findall(r'%\w+', mlir_text))
        numbered = len(re.findall(r'%\d+', mlir_text))
        return f"""
  SSA 值命名:
    两种格式:
      %pid          ← 显式名称 (有意义，用于重要的值)
      %0, %1, %2    ← 数字名称 (自动分配，用于中间值)

    在 TTIR 中:
      命名值: {named} 个
      编号值: {numbered} 个

    规则:
      • 每个值只能被定义一次 (SSA 约束)
      • 使用前必须先定义 (支配关系)
      • 名称只在当前 Region 内可见 (作用域)
  """

    @staticmethod
    def explain_operation_syntax():
        """解释 operation 的完整语法。"""
        return """
  Operation 完整语法 (BNF 风格):

    operation ::= (ssa-use-list "=")? op-name operands
                  ("{" attribute-dict "}")?
                  (":" type-list)?
                  location?

    ssa-use-list ::= ssa-use ("," ssa-use)*
    op-name       ::= (dialect ".")? bare-id
    operands      ::= "(" (operand ("," operand)*)? ")"
    operand       ::= ssa-use ":" type
    type-list     ::= functional-type | type
    functional-type ::= "(" type-list ")" "->" type-list

  实例分解:

    %z = arith.addf %x, %y : f32
    │   │      │    │   │    │
    │   │      │    │   │    └─── 结果类型 (scalar f32)
    │   │      │    │   └───────── operand (引用 %y)
    │   │      │    └───────────── operand (引用 %x)
    │   │      └─────────────────── op 名 (arith dialect, addf op)
    │   └─────────────────────────── 等号前: 定义的结果
    └──────────────────────────────── result SSA 值名

    Tensor 版本:

      %z = arith.addf %x, %y : tensor<256xf32>
      │                              │
      │                              └─── tensor 类型 (不是 scalar!)
      └─────────────────────────────────── 结果也是 tensor

    Op 没有 result 的情况 (如 tt.store):
      tt.store %ptr, %val, %mask : tensor<256x!tt.ptr<f32>>
      │
      └── 没有 %result = 前缀 → 这是一个 side-effect op
  """

    @staticmethod
    def explain_type_syntax():
        """解释类型语法。"""
        return """
  MLIR 类型语法:

  ┌──────────────────────────────────┬──────────────────────────────┐
  │ 语法                              │ 含义                          │
  ├──────────────────────────────────┼──────────────────────────────┤
  │ i32, f32, f16, f64             │ 标准标量类型                   │
  │ i1                              │ 布尔 (用于 mask)              │
  │ index                           │ 平台相关的整数 (用于索引)      │
  │ tensor<NxMxf32>                 │ N×M 的 f32 tensor            │
  │ tensor<...xT, #layout>          │ 带 layout attribute 的 tensor │
  │ !tt.ptr<f32>                    │ Triton 指针 (自定义类型!)     │
  │ !tt.ptr<tensor<256xf32>>        │ 指向 tensor 的指针 (罕见)      │
  │ memref<Nxf32>                   │ 内存引用 (类似 buffer)         │
  │ vector<Nxf32>                   │ 向量类型 (SIMD)               │
  └──────────────────────────────────┴──────────────────────────────┘

  关键区分:
    tensor<256xf32>     ← 256 个 f32 的逻辑 tensor (在 computation 中使用)
    !tt.ptr<f32>        ← 指向 f32 的指针 (在寻址中使用)
    memref<256xf32>     ← 256 个 f32 的内存区域 (带 shape+stride)

    在 Triton 中主要看到: i32, f32, tensor<>, !tt.ptr<>
  """

    @staticmethod
    def explain_attribute_syntax():
        """解释 attribute 语法。"""
        return """
  MLIR Attribute 语法:

  1. 内置 attribute (无 # 前缀):
     {key = value, ...} 形式 (attr-dict):
       {tt.divisibility = 16 : i32, noinline = false}

  2. Dialect attribute (有 # 前缀):
     #dialect.attr_name<{parameters}>
       #ttg.blocked<{sizePerThread=[1], threadsPerWarp=[32], ...}>

  3. 特殊 attribute:
     #loc             ← 位置信息 (调试用)
     #loc1 = loc(...) ← 位置别名

  Attributes vs Operands 的分辨方法:
    attr 在 {{}} 内, 或 # 前缀   → 编译期常量
    operand 在 () 内, % 前缀     → 运行时 SSA 值
  """

    @staticmethod
    def explain_region_and_block():
        """解释 Region 和 Block 语法。"""
        return """
  Region 和 Block 的结构:

  单个 Region (最常见):
    tt.func @name(...) {
      ^bb0(%arg0: i32, %arg1: f32):    ← Block 标签 (可省略)
        %0 = arith.constant 42 : i32    ← Operation 1
        %1 = arith.addi %arg0, %0       ← Operation 2
        tt.return                        ← Terminator
    }

  多个 Block (有分支时):
    scf.if %cond -> f32 {
      ^bb0:                              ← true branch
        %0 = arith.addf %a, %b
        scf.yield %0
    } else {
      ^bb1:                              ← false branch
        %1 = arith.subf %a, %b
        scf.yield %1
    }

  规则:
    • 每个 Region 至少有一个 Block
    • 每个 Block 以 Terminator (如 tt.return, scf.yield) 结束
    • Block 可以有参数 (如 ^bb0(%arg0:i32))
    • Block 参数是"phi node 替代品" (MLIR 不需要 phi)
  """


# ══════════════════════════════════════════════════════════════════════
# 实战: 手工分析 MLIR
# ══════════════════════════════════════════════════════════════════════


def analyze_ssa_use_def(mlir_text):
    """分析 SSA use-def chain。"""
    # 找所有定义
    defs = {}
    for m in re.finditer(r'(%\w+)\s*=', mlir_text):
        name = m.group(1)
        # 找这个值被使用的位置
        uses = []
        for m2 in re.finditer(rf'{re.escape(name)}(?!\s*=)', mlir_text):
            uses.append(m2.start())
        defs[name] = {
            "line_defined": mlir_text[:m.start()].count("\n") + 1,
            "num_uses": len(uses),
        }
    return defs


def locate_op_in_text(mlir_text, op_name):
    """在 MLIR 文本中定位特定 op。"""
    lines = mlir_text.split("\n")
    results = []
    for i, line in enumerate(lines):
        if op_name in line:
            results.append((i + 1, line.strip()))
    return results


# ══════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════


def main():
    print("=" * 70)
    print("  23 — MLIR 文本格式语法详解")
    print("=" * 70)

    # 生成 IR
    N = 256
    x = torch.randn(N, device="cuda")
    out = torch.empty(N, device="cuda")
    format_demo[(1,)](x, out, N, BLOCK=256)
    torch.cuda.synchronize()

    ttir_text = get_fresh_ttir()
    if not ttir_text:
        print("  ⚠ 未找到 TTIR 文件")
        return

    explainer = MLIRSyntaxExplainer()

    # ── 1. Module 结构 ─────────────────────────────────────
    print("─" * 70)
    print("  1. Module (顶层容器)")
    print("─" * 70)
    print(explainer.explain_module_header(ttir_text))

    # ── 2. Function 签名 ──────────────────────────────────
    print("─" * 70)
    print("  2. Function (函数)")
    print("─" * 70)
    print(explainer.explain_function_signature(ttir_text))

    # ── 3. SSA 命名 ───────────────────────────────────────
    print("─" * 70)
    print("  3. SSA 值命名和使用")
    print("─" * 70)
    print(explainer.explain_ssa_naming(ttir_text))

    # 分析 use-def chain
    def_analysis = analyze_ssa_use_def(ttir_text)
    print("  Use-Def 分析 (前 10 个):")
    for name, info in list(def_analysis.items())[:10]:
        print(f"    {name}: 定义于行 {info['line_defined']}, 使用了 {info['num_uses']} 次")
    unused = [name for name, info in def_analysis.items() if info['num_uses'] == 0]
    if unused:
        print(f"\n  ⚠ 未使用的值: {unused}")
        print(f"    (这些可能是编译器优化后的死代码，或 store 类 op 的间接使用)")

    # ── 4. Operation 语法 ─────────────────────────────────
    print("\n" + "─" * 70)
    print("  4. Operation 语法详解")
    print("─" * 70)
    print(explainer.explain_operation_syntax())

    # 在真实 TTIR 中定位关键 op
    print("  在真实 TTIR 中定位关键 op:")
    for op_name in ["tt.get_program_id", "tt.make_range", "arith.addi",
                     "tt.splat", "tt.load", "arith.addf", "tt.store"]:
        locs = locate_op_in_text(ttir_text, op_name)
        if locs:
            print(f"    {op_name}: 第 {locs[0][0]} 行")
            print(f"      {locs[0][1][:120]}")

    # ── 5. Type 语法 ──────────────────────────────────────
    print("\n" + "─" * 70)
    print("  5. Type 语法详解")
    print("─" * 70)
    print(explainer.explain_type_syntax())

    # ── 6. Attribute 语法 ─────────────────────────────────
    print("─" * 70)
    print("  6. Attribute 语法详解")
    print("─" * 70)
    print(explainer.explain_attribute_syntax())

    # ── 7. Region 和 Block ────────────────────────────────
    print("─" * 70)
    print("  7. Region, Block, 和 Terminator")
    print("─" * 70)
    print(explainer.explain_region_and_block())

    # 在真实 TTIR 中展示嵌套
    print("  真实 TTIR 中的结构层次:")
    # 显示 module → function → body 的缩进结构
    lines = ttir_text.split("\n")
    for line in lines[:30]:
        indent = len(line) - len(line.lstrip())
        if "module" in line or "tt.func" in line or "return" in line:
            print(f"    {'  ' * (indent // 2)}{line.strip()[:120]}")

    # ── 8. 手工构造 MLIR ──────────────────────────────────
    print("\n" + "─" * 70)
    print("  8. 手工构造简单 MLIR (练习)")
    print("─" * 70)
    print("""
  假设你要手工写一个 vector add kernel 的 TTIR:

  ```mlir
  module {
    tt.func public @my_add(
      %x: !tt.ptr<f32>,
      %y: !tt.ptr<f32>,
      %out: !tt.ptr<f32>,
      %N: i32
    ) {
      // 1. 获取 block id
      %pid = tt.get_program_id x : i32

      // 2. 生成偏移 [0, 1, 2, ..., 255]
      %c256 = arith.constant 256 : i32
      %block_start = arith.muli %pid, %c256 : i32
      %offsets = tt.make_range {start = 0 : i32, end = 256 : i32}
        : tensor<256xi32>
      %0 = tt.splat %block_start : i32 -> tensor<256xi32>
      %global_offsets = arith.addi %0, %offsets : tensor<256xi32>

      // 3. 边界检查
      %n_splat = tt.splat %N : i32 -> tensor<256xi32>
      %mask = arith.cmpi slt, %global_offsets, %n_splat : tensor<256xi32>

      // 4. Load
      %x_ptr_splat = tt.splat %x : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
      %x_addrs = tt.addptr %x_ptr_splat, %global_offsets
        : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
      %x_vals = tt.load %x_addrs, %mask : tensor<256x!tt.ptr<f32>>

      %y_ptr_splat = tt.splat %y : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
      %y_addrs = tt.addptr %y_ptr_splat, %global_offsets
        : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
      %y_vals = tt.load %y_addrs, %mask : tensor<256x!tt.ptr<f32>>

      // 5. Compute
      %result = arith.addf %x_vals, %y_vals : tensor<256xf32>

      // 6. Store
      %out_ptr_splat = tt.splat %out : !tt.ptr<f32>
        -> tensor<256x!tt.ptr<f32>>
      %out_addrs = tt.addptr %out_ptr_splat, %global_offsets
        : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
      tt.store %out_addrs, %result, %mask
        : tensor<256x!tt.ptr<f32>>

      tt.return
    }
  }
  ```

  关键模式:
    • 指针运算: splat ptr → addptr → load/store
    • tensor 构造: make_range → arith constant → splat → addi
    • 边界检查: splat N → cmpi → 用作 load/store 的 mask
    • 每个 op 都跟上 ': types' 标注

  🔑 手工构造 MLIR 的价值:
    • 帮助你精确理解每步编译做了什么
    • 在 debug 时，你能认出"不对"的 IR pattern
    • 未来可以直接给编译器喂手工优化的 IR
""")

    print("\n📖 下一步: python phase4_compiler/24_triton_tt_dialect.py")
    print("   Triton 的 tt dialect 完整参考——每个 op 的语义和用法。\n")


if __name__ == "__main__":
    main()
