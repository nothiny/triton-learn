"""
自定义 MLIR Pass 的 Python 绑定框架

Triton 的编译管线基于 MLIR。虽然大部分用户不需要自定义 pass，
但了解如何注册 pass 对于深入理解有帮助。

注意：Triton 的 pass 注册机制与 triton 版本有关，
     此文件是概念性框架，实际使用需要参考对应版本的 API。

TODO: 根据 Triton 版本调整具体的 pass 注册方式
"""

# ---------------------------------------------------------------------------
# 概念：注册一个自定义 pass
# ---------------------------------------------------------------------------

# Triton passes are MLIR passes registered in C++ with Python bindings.
# The typical pattern:
#
# 1. Define the pass in C++ (using MLIR's Pass base classes)
# 2. Register it in Triton's pass pipeline
# 3. Choose when it runs (before/after specific stages)

# For learning purposes, we outline what a simple counting pass would do:
"""
class MyCountOpsPass(MLIRPass):
    '''
    一个简单的计数 pass: 遍历 IR 中的每个 op，统计类型分布。

    这类似于编译器中的 IR statistics 收集。
    '''
    def run(self, module):
        op_counts = {}
        for op in module.body.operations:
            op_name = op.name
            op_counts[op_name] = op_counts.get(op_name, 0) + 1
        print(f"Op distribution: {op_counts}")
        return module
"""

# ---------------------------------------------------------------------------
# 替代方案：使用 Python AST 分析 Triton kernel 源码
# ---------------------------------------------------------------------------


def analyze_kernel_ast(kernel_fn):
    """
    使用 Python AST 分析 Triton kernel 的结构，
    作为编译器分析的 lightweight 替代。

    Args:
        kernel_fn: @triton.jit 装饰的函数

    Returns:
        dict: 包含 op 计数、循环深度等信息
    """
    import ast, inspect

    src = inspect.getsource(kernel_fn)
    tree = ast.parse(src)

    stats = {
        "loads": 0,
        "stores": 0,
        "dots": 0,
        "reductions": 0,
        "loops": 0,
        "max_loop_depth": 0,
        "constexpr_params": [],
    }

    class Analyzer(ast.NodeVisitor):
        def __init__(self):
            self.loop_depth = 0

        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == "load":
                    stats["loads"] += 1
                elif node.func.attr == "store":
                    stats["stores"] += 1
                elif node.func.attr == "dot":
                    stats["dots"] += 1
                elif node.func.attr in ("sum", "max", "min"):
                    stats["reductions"] += 1
            self.generic_visit(node)

        def visit_For(self, node):
            stats["loops"] += 1
            self.loop_depth += 1
            stats["max_loop_depth"] = max(stats["max_loop_depth"], self.loop_depth)
            self.generic_visit(node)
            self.loop_depth -= 1

    Analyzer().visit(tree)
    return stats


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import triton
    import triton.language as tl

    @triton.jit
    def example_kernel(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        z = x + y
        tl.store(y_ptr + offs, z, mask=mask)

    stats = analyze_kernel_ast(example_kernel.fn)
    print("Kernel statistics (from Python AST):")
    for k, v in stats.items():
        print(f"  {k}: {v}")
