"""
13_custom_pass.py — 自定义 MLIR Pass: 扩展编译器

学习目标:
  1. 理解什么是 MLIR Pass
  2. 了解如何在 Python 中注册和分析 pass
  3. 构建一个实用的 kernel 分析工具（基于 AST）

运行: python phase4_compiler/13_custom_pass.py

前提: 已运行 01-12，对整个编译管线有完整理解。
注意: Triton 的 Python pass API 在不同版本间变化较大。
      这个文件聚焦于概念理解和可行的分析工具。
"""

import ast
import inspect
from typing import Any


# ══════════════════════════════════════════════════════════════════════
# 第一部分: MLIR Pass 概念
# ══════════════════════════════════════════════════════════════════════


def explain_pass_concept():
    """讲解什么是 MLIR Pass。"""
    print("─" * 70)
    print("  1. 什么是 MLIR Pass?")
    print("─" * 70)
    print("""
  MLIR (Multi-Level Intermediate Representation) 是一个编译器框架。
  它允许定义多种 "dialect" (方言)，每种方言有自己的 op 和类型。

  Pass = 对 MLIR module 做一次变换的函数。

  输入: MLIR module (当前状态的 IR)
  输出: MLIR module (变换后的 IR)
  约束: 不能破坏 IR 的语义

  例子:
    • Dead Code Elimination pass: 删除不会被执行到的代码
    • Constant Folding pass:    在编译期算出常量表达式的结果
    • Inlining pass:            把函数调用替换为函数体
    • ConvertLayout pass:       改变 tensor 的 layout encoding

  Triton 的每个 pass 都是 MLIR pass。
  但 Triton 用 C++ 写 pass (在 lib/Conversion/ 和 lib/Dialect/ 下)，
  然后通过 pybind11 暴露给 Python。""")

    # ── Pass 的结构 ──
    print("""
  ──────────────────────────────────────────────────────────────────
  Pass 的基本结构 (概念性)
  ──────────────────────────────────────────────────────────────────

  C++ 侧 (实际代码在 triton 源码中):
    class MyPass : public mlir::PassWrapper<MyPass, ...> {
      void runOnOperation() override {
        // 遍历 module 中的所有 op
        getOperation()->walk([&](mlir::Operation *op) {
          if (op->getName() == "tt.dot") {
            // 对每个 tt.dot 做点什么
            // 例如: 替换为 MMA intrinsic
          }
        });
      }
    };

  Python 侧 (Triton 提供的绑定):
    @passes.register_pass("my_pass")
    def my_pass(module):
        # module 是 MLIR module 的 Python wrapper
        for op in module.body.operations:
            print(op.name)
        return module  # 如果只是分析，返回原 module

  但是, Triton 的 Python pass API 在不同版本间有差异。
  最稳定的方法是从 Python AST 层面分析 kernel。""")


# ══════════════════════════════════════════════════════════════════════
# 第二部分: Python AST 分析工具 (稳定可用)
# ══════════════════════════════════════════════════════════════════════


class KernelASTAnalyzer(ast.NodeVisitor):
    """
    分析 @triton.jit kernel 的 Python AST。
    虽然不是真正的 compiler pass，但可以提供有价值的 insights。

    统计:
      - tl.load / tl.store 的数量
      - tl.dot 的数量
      - tl.sum / tl.max 等的数量
      - 循环层数
      - constexpr 参数
    """

    def __init__(self):
        self.stats = {
            "num_loads": 0,
            "num_stores": 0,
            "num_dots": 0,
            "num_reductions": 0,
            "num_broadcasts": 0,
            "num_for_loops": 0,
            "max_loop_depth": 0,
            "num_ifs": 0,
            "constexpr_params": [],
        }
        self._loop_depth = 0

    def visit_arguments(self, node: ast.arguments):
        """收集 constexpr 参数。"""
        for arg in node.args:
            if arg.annotation:
                # 检查是否是 tl.constexpr
                if isinstance(arg.annotation, ast.Attribute):
                    if arg.annotation.attr == "constexpr":
                        self.stats["constexpr_params"].append(arg.arg)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        """识别 tl.* 调用。"""
        # 检查 func 是否是 tl.xxx 形式
        if isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            if attr == "load":
                self.stats["num_loads"] += 1
            elif attr == "store":
                self.stats["num_stores"] += 1
            elif attr == "dot":
                self.stats["num_dots"] += 1
            elif attr in ("sum", "max", "min", "argmax", "argmin"):
                self.stats["num_reductions"] += 1
            elif attr == "broadcast_to":
                self.stats["num_broadcasts"] += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        self.stats["num_for_loops"] += 1
        self._loop_depth += 1
        self.stats["max_loop_depth"] = max(
            self.stats["max_loop_depth"], self._loop_depth
        )
        self.generic_visit(node)
        self._loop_depth -= 1

    def visit_If(self, node: ast.If):
        self.stats["num_ifs"] += 1
        self.generic_visit(node)


def analyze_kernel(kernel_fn) -> dict[str, Any]:
    """
    分析一个 @triton.jit 函数的 AST。

    Args:
        kernel_fn: 被 @triton.jit 装饰的函数

    Returns:
        dict: 统计信息
    """
    import triton

    import textwrap
    if hasattr(kernel_fn, 'fn'):
        # JITFunction → 取原始 Python 函数
        src = inspect.getsource(kernel_fn.fn)
    else:
        src = inspect.getsource(kernel_fn)

    src = textwrap.dedent(src)  # remove common leading whitespace
    tree = ast.parse(src)
    analyzer = KernelASTAnalyzer()
    analyzer.visit(tree)
    return analyzer.stats


def estimate_arith_intensity(stats: dict[str, Any],
                               block_m: int, block_n: int, block_k: int = 16) -> dict:
    """
    从 AST 统计估算 kernel 的算术强度。

    Args:
        stats: analyze_kernel 的输出
        block_m, block_n, block_k: tile 大小

    Returns:
        dict: 包含估算的 FLOPs, bytes, 和算术强度
    """
    # 每个 load 读取 block_m × block_n 个元素
    dtype_size = 2  # 假设 fp16
    bytes_per_load = block_m * block_n * dtype_size
    bytes_per_store = block_m * block_n * dtype_size

    total_bytes = (stats["num_loads"] * bytes_per_load +
                   stats["num_stores"] * bytes_per_store)

    # 每个 dot: 2×M×N×K FLOPs
    flops_per_dot = 2 * block_m * block_n * block_k
    total_flops = stats["num_dots"] * flops_per_dot

    # elementwise (粗略: 每个元素 1 FLOP)
    total_flops += stats["num_loads"] * block_m * block_n

    ai = total_flops / total_bytes if total_bytes > 0 else float("inf")

    return {
        "estimated_flops": total_flops,
        "estimated_bytes": total_bytes,
        "estimated_ai": ai,
    }


# ══════════════════════════════════════════════════════════════════════
# 第三部分: 演示
# ══════════════════════════════════════════════════════════════════════


def demo_analyze_kernels():
    """用几个示例 kernel 演示 AST 分析。"""
    import triton
    import triton.language as tl

    print("\n" + "─" * 70)
    print("  2. 演示: 用 AST 分析器分析 kernel")
    print("─" * 70)

    @triton.jit
    def simple_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
        """简单的 elementwise kernel。"""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)

    @triton.jit
    def complex_kernel(A, B, C, M, N, K,
                        BLOCK_M: tl.constexpr,
                        BLOCK_N: tl.constexpr,
                        BLOCK_K: tl.constexpr):
        """较复杂的 matmul kernel。"""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        A_ptr = A + rm[:, None] * K + rk[None, :]
        B_ptr = B + rk[:, None] * N + rn[None, :]
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a = tl.load(A_ptr + k)
            b = tl.load(B_ptr + k * N)
            acc += tl.dot(a, b)
            if k > 0:
                acc = acc * 0.99
        tl.store(C + rm[:, None] * N + rn[None, :], acc)

    for name, kernel in [("simple_kernel", simple_kernel),
                           ("complex_kernel", complex_kernel)]:
        print(f"\n  ▸ {name}")
        stats = analyze_kernel(kernel)
        print(f"    Loads:     {stats['num_loads']}")
        print(f"    Stores:    {stats['num_stores']}")
        print(f"    Dots:      {stats['num_dots']}")
        print(f"    Reductions:{stats['num_reductions']}")
        print(f"    For loops: {stats['num_for_loops']}")
        print(f"    Max depth: {stats['max_loop_depth']}")
        print(f"    Ifs:       {stats['num_ifs']}")
        print(f"    Constexpr: {stats['constexpr_params']}")

        # 估算算术强度
        if stats["num_dots"] > 0:
            est = estimate_arith_intensity(stats, block_m=64, block_n=64, block_k=32)
            print(f"    估算 FLOPs:  {est['estimated_flops']:,}")
            print(f"    估算 Bytes:  {est['estimated_bytes']:,}")
            print(f"    估算 AI:     {est['estimated_ai']:.2f} FLOP/byte")


# ══════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════


def main():
    print("=" * 70)
    print("  13 — 自定义 MLIR Pass 与 AST 分析")
    print("=" * 70)

    # 概念讲解
    explain_pass_concept()

    # 演示 AST 分析
    demo_analyze_kernels()

    # ── 展望 ──
    print("\n" + "─" * 70)
    print("  3. 展望: 真正的自定义 Pass")
    print("─" * 70)
    print("""
  Triton 的 Python pass API 在 Triton 3.x 中仍在发展。

  当前你可以做的:
    ✅ Python AST 分析 (本文件演示的)
    ✅ 环境变量 dump IR + grep 分析
    ✅ 对比不同 config 的 PTX

  未来 (Triton 后续版本):
    🔄 更稳定的 Python pass 注册 API
    🔄 更丰富的 IR 自省工具
    🔄 更好的 programmatic compile API

  如果你需要真正的自定义 pass:
    1. Fork triton 源码
    2. 在 lib/Conversion/ 下写 C++ pass
    3. 在 Python 端绑定你的 pass
    → 参考: https://github.com/triton-lang/triton/blob/main/lib/Conversion/

  对于大多数用户:
    理解 compiler pipeline + 会 dump IR + 会读 PTX
    → 已经足够调试和优化绝大多数性能问题了。""")

    # ── Phase 4 总结 ──
    print("\n" + "=" * 70)
    print("  Phase 4 总结: 你学到了什么")
    print("=" * 70)
    print("""
  ✅ 01: 第一次接触 4 层 IR (TTIR → TTGIR → LLVM → PTX)
  ✅ 02: TTIR 的每个 op 对应什么 Python 代码
  ✅ 03: TTGIR 中 layout encoding 的出现和含义
  ✅ 04: 5 种 layout 类型的深入理解
  ✅ 05: Layout conversion 的代价 — 隐形的性能杀手
  ✅ 06: LLVM IR 的寄存器、地址计算、NVVM intrinsic
  ✅ 07: PTX 汇编精读 — GPU 真正执行的指令
  ✅ 08: Pass pipeline 全景 — 每个 pass 做什么
  ✅ 09: Software pipelining — 让加载和计算重叠
  ✅ 10: 寄存器分配和三资源约束
  ✅ 11: 实战诊断 4 种常见性能问题
  ✅ 12: triton.compiler API — 程序化访问编译管线
  ✅ 13: 自定义 pass 概念和 AST 分析工具

  🎯 核心收获:
    1. 编译器管线的每一层都"丢掉一些信息，加入一些信息"
    2. Layout encoding 是 Triton 最独特的设计 — 自动化线程→数据映射
    3. 能读懂 PTX = 能诊断大多数性能问题
    4. 环境变量是你的朋友: TRITON_KERNEL_DUMP, MLIR_PRINT_IR_AFTER_ALL

  📚 继续学习:
    • Triton 源码: lib/Conversion/, lib/Dialect/
    • MLIR 官方文档: https://mlir.llvm.org/docs/
    • Triton 论文: https://www.eecs.harvard.edu/~htk/publication/2023-pact-tillet-kung-cox.pdf
    • NVIDIA PTX ISA 手册: https://docs.nvidia.com/cuda/parallel-thread-execution/
""")

    print("\n🏁 Phase 4 完成！\n")


if __name__ == "__main__":
    main()
