"""
20_source_guide.py — Triton 源码导航

学习目标:
  1. 了解 Triton 源码的整体结构
  2. 知道每个关键文件/模块做了什么
  3. 建立"遇到问题 → 找到对应源码 → 理解机制"的能力

运行: python phase4_compiler/20_source_guide.py

前提: 已完成 01-19。
"""

import os
import sys
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════


def main():
    print("=" * 70)
    print("  20 — Triton 源码导航")
    print("=" * 70)

    # ── 找到 Triton 安装路径 ──────────────────────────────
    import triton
    triton_root = Path(triton.__path__[0])
    print(f"\n  Triton 安装路径: {triton_root}")

    # ── Python 端源码结构 ─────────────────────────────────
    print("\n" + "─" * 70)
    print("  Python 端源码结构 (triton/python/triton/)")
    print("─" * 70)

    PYTHON_SOURCE_TREE = {
        "python/triton/": {
            "_description": "Triton 的 Python 层入口",
            "runtime/": {
                "jit.py": "★★★ JITFunction 类 — @triton.jit 的实现\n"
                         "• JITFunction.__call__: kernel 调用的入口\n"
                         "• cache_key 计算\n"
                         "• 编译调度 (触发 compile 还是从 cache 加载)",
                "driver.py": "★★ CUDA driver 交互 — cuLaunchKernel 的 Python 封装",
                "cache.py": "★★ JIT 缓存管理 — 读写 ~/.triton/cache/",
                "autotuner.py": "★★ Autotuner 实现 — 搜索 config → 测时 → 缓存",
            },
            "language/": {
                "_description": "★★★ triton.language 模块 — 你写的每一个 tl.* 都在这里定义",
                "__init__.py": "tl.load, tl.store, tl.dot, tl.arange 等 API 定义\n"
                              "• 每个函数都是 JITFunction 内部的 '语义标记'\n"
                              "• 在 JIT 模式下返回 TensorDescriptor\n"
                              "• 在解释器模式下返回真实的 torch tensor",
                "semantic.py": "★★★ AST 语义分析器 — 遍历 Python AST → 生成 TTIR builder\n"
                              "• visit_Call: 识别 tl.load/tl.store/tl.dot\n"
                              "• 构建 MLIR operations\n"
                              "• 类型推导和检查",
                "core.py": "★★ 核心类型: tensor, pointer, block, scalar\n"
                          "• TensorDescriptor: 编译期 tensor 占位符\n"
                          "• dtype 和 shape 的编译期表示",
            },
            "compiler/": {
                "_description": "★★★ 编译器入口和调度",
                "compiler.py": "★★ compile() 入口 (TTIR→TTGIR→LLVM→PTX 的调度)",
                "code_generator.py": "★★ AST → TTIR MLIR 的生成器",
            },
            "tools/": {
                "_description": "工具和实用脚本",
                "compile.py": "命令行编译工具",
            },
        }
    }

    def print_tree(tree, indent=0):
        for key, value in tree.items():
            if key.startswith("_"):
                print(f"{'  ' * indent}{key[1:]}")
                if isinstance(value, str):
                    for line in value.strip().split("\n"):
                        print(f"{'  ' * indent}  {line}")
            elif isinstance(value, dict):
                print(f"{'  ' * indent}{key}")
                print_tree(value, indent + 1)
            elif isinstance(value, str):
                print(f"{'  ' * indent}{key}")
                for line in value.strip().split("\n"):
                    print(f"{'  ' * indent}  {line}")

    print_tree(PYTHON_SOURCE_TREE)

    # ── C++ 端源码结构 ─────────────────────────────────────
    print("\n" + "─" * 70)
    print("  C++ 端源码结构 (triton/lib/)")
    print("─" * 70)

    CPP_SOURCE_TREE = {
        "lib/": {
            "_description": "Triton 的 C++ 层 — MLIR passes 和 dialect 定义",
            "Conversion/": {
                "_description": "★★★ MLIR Conversion Passes — 最关键的部分",
                "TritonToTritonGPU/": "★★★ ConvertTritonToTritonGPU pass\n"
                    "• 分配 layout encoding\n"
                    "• 决定 sizePerThread, threadsPerWarp, warpsPerCTA\n"
                    "• 插入 convert_layout",
                "TritonGPUToLLVM/": "★★ ConvertTritonGPUToLLVM pass\n"
                    "• 展开 layout encoding → 线程索引计算\n"
                    "• 生成 NVVM intrinsic\n"
                    "• tt.dot → LLVM MMA intrinsic",
            },
            "Dialect/": {
                "_description": "★★ MLIR Dialect 定义",
                "Triton/": "tt dialect: tt.load, tt.store, tt.dot, tt.reduce",
                "TritonGPU/": "ttg dialect: layout encoding types, convert_layout, async_copy",
            },
            "Analysis/": {
                "_description": "★★ 编译器分析",
                "LayoutAnalysis.cpp": "Layout 分配算法",
                "Utility.cpp": "多种分析工具",
            },
            "Transform/": {
                "_description": "★★★ Optimization Passes",
                "AccelerateMatmul.cpp": "tt.dot → MMA intrinsic",
                "Pipeline.cpp": "★★ Software pipelining (循环展开 + cp.async)",
                "Prefetch.cpp": "★★ Prefetch 插入",
                "Coalesce.cpp": "Coalesced access 优化",
                "RemoveLayoutConversions.cpp": "消除冗余 convert_layout",
                "Combine.cpp": "Op 融合 (如 add+relu)",
            },
        }
    }

    print_tree(CPP_SOURCE_TREE)

    # ── 建议的阅读顺序 ─────────────────────────────────────
    print("\n" + "─" * 70)
    print("  建议的阅读顺序 (按你想解决的问题)")
    print("─" * 70)
    print("""
  ┌────────────────────────┬──────────────────────────────────────────┐
  │ 想了解什么               │ 阅读顺序                                  │
  ├────────────────────────┼──────────────────────────────────────────┤
  │ @triton.jit 怎么工作    │ jit.py → semantic.py → code_generator.py │
  │ tl.load 怎么变 load 指令 │ semantic.py → TritonToTritonGPU/         │
  │                           → TritonGPUToLLVM/                       │
  │ tl.dot 怎么变 MMA       │ semantic.py → AccelerateMatmul.cpp       │
  │                           → TritonGPUToLLVM/                       │
  │ Pipeline 怎么展开循环     │ Pipeline.cpp → Prefetch.cpp             │
  │ Layout 怎么分配          │ TritonToTritonGPU/ → LayoutAnalysis.cpp  │
  │ Autotune 怎么搜索         │ autotuner.py → cache.py                 │
  │ CUDA driver 怎么交互     │ driver.py → jit.py (底层调用)            │
  └────────────────────────┴──────────────────────────────────────────┘

  🔑 最重要的 3 个文件:
    1. python/triton/language/semantic.py
       → 理解你的 Python 代码怎么变成 IR builder

    2. lib/Conversion/TritonToTritonGPU/
       → 理解 layout encoding 怎么分配

    3. lib/Transform/AccelerateMatmul.cpp
       → 理解 tl.dot 怎么变成 MMA 指令
""")

    # ── 如何阅读源码 ───────────────────────────────────────
    print("─" * 70)
    print("  如何阅读 Triton 源码")
    print("─" * 70)
    print("""
  Triton 的 GitHub: https://github.com/triton-lang/triton

  阅读技巧:
    1. 带问题阅读: 不要通读，而是"我想知道 X 是怎么实现的"
    2. 从 Python 层开始: Python 代码更好读，也是你日常使用的接口
    3. 顺着调用链读: semantic.py → code_generator.py → Conversion passes
    4. 用 grep 找关键函数: git grep "ConvertTritonToTritonGPU"
    5. 看测试: test/ 目录下的测试是最好的"使用文档"

  设置本地开发环境:
    git clone https://github.com/triton-lang/triton.git
    cd triton
    pip install -e ".[dev]"

    # 编译 C++ 部分 (需要 CMake + LLVM)
    python setup.py build

    # 运行测试确保一切正常
    pytest python/test/unit/
""")

    # ── 贡献代码的入口点 ──────────────────────────────────
    print("─" * 70)
    print("  常见修改的入口点")
    print("─" * 70)
    print("""
  如果你要...
    添加新的 tl.* op:
      → python/triton/language/__init__.py (定义 Python API)
      → python/triton/language/semantic.py (添加 AST visitor)
      → lib/Dialect/Triton/ (定义 MLIR op)
      → lib/Conversion/TritonToTritonGPU/ (添加 lowering)

    添加新的 MLIR pass:
      → lib/Transform/YourPass.cpp
      → 注册到 lib/Conversion/ 的 pass pipeline

    修改 autotuner 行为:
      → python/triton/runtime/autotuner.py

    修改 cache 策略:
      → python/triton/runtime/cache.py

    添加对新 GPU 架构的支持:
      → lib/Dialect/TritonGPU/ (新的 MMA 形状等)
      → lib/Conversion/TritonGPUToLLVM/ (新的 PTX intrinsic)
""")

    print("\n📖 下一步: python phase4_compiler/21_ptx_to_sass.py")
    print("   深入 PTX → SASS: GPU 真正执行的机器码。\n")


if __name__ == "__main__":
    main()
