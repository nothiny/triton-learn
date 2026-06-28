"""
12_compile_api.py — triton.compiler API: 程序化访问编译管线

学习目标:
  1. 了解 triton.compiler 模块的 Python API
  2. 学会不运行 kernel 就获取 IR
  3. 理解 JIT 编译的 cache 机制

运行: python phase4_compiler/12_compile_api.py

前提: 已运行 01-11，对整个编译管线有完整理解。
"""

import os
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


# ══════════════════════════════════════════════════════════════════════
# 一个简单的 kernel 用于测试 API
# ══════════════════════════════════════════════════════════════════════


@triton.jit
def api_test_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
    """最简单的 kernel，用于测试编译器 API。"""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


# ══════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════


def main():
    print("=" * 70)
    print("  12 — triton.compiler API: 程序化访问编译管线")
    print("=" * 70)

    # ── 1. triton.compiler 模块结构 ──────────────────────────
    print("─" * 70)
    print("  1. triton.compiler 模块结构")
    print("─" * 70)

    try:
        import triton.compiler as compiler
        print(f"  triton.compiler 版本: {getattr(compiler, '__version__', 'unknown')}")
        print(f"  可用属性: {[x for x in dir(compiler) if not x.startswith('_')][:15]}")
    except ImportError as e:
        print(f"  triton.compiler 不可用: {e}")

    print("""
  triton.compiler 模块的核心组件:
    • triton.compiler.compile()    — 编译 kernel (如果 API 暴露)
    • triton.compiler.ASTSource     — kernel 的 AST 表示
    • triton.compiler.CompiledKernel — 编译产物的封装
    • triton.compiler.make_backend  — 选择编译后端 (CUDA/HIP/...)

  ⚠ API 稳定性: triton.compiler 的 Python API 在不同 Triton 版本间
     变化较大。截止 Triton 3.x, 最可靠的方式仍然是通过环境变量
     TRITON_KERNEL_DUMP=1 来获取 IR。""")

    # ── 2. JITFunction 对象 ─────────────────────────────────
    print("─" * 70)
    print("  2. JITFunction — @triton.jit 的内涵")
    print("─" * 70)

    # api_test_kernel 是一个 JITFunction 实例
    print(f"  type(api_test_kernel) = {type(api_test_kernel)}")
    print(f"  api_test_kernel.__name__ = {api_test_kernel.__name__}")

    # 检查 JITFunction 的属性
    jit_attrs = [a for a in dir(api_test_kernel) if not a.startswith('_')]
    print(f"  JITFunction 可用属性: {jit_attrs}")

    # 看看有没有 fn (原始函数) 和 cache 相关信息
    if hasattr(api_test_kernel, 'fn'):
        print(f"  原始函数: {api_test_kernel.fn}")
    if hasattr(api_test_kernel, 'cache_key'):
        print(f"  Cache key: {api_test_kernel.cache_key}")

    print("""
  @triton.jit 装饰器做的事情:
    1. 解析被装饰函数的 Python AST
    2. 识别 tl.constexpr 参数 (编译期常量)
    3. 识别 tl.* 调用 (load, store, dot, ...)
    4. 生成 TTIR 构建函数
    5. 包装为一个 JITFunction 对象

    当你调用 kernel[grid](args...) 时:
    1. 计算 cache key (源码 hash + constexpr values + GPU arch + Triton version)
    2. 查 ~/.triton/cache/ 有没有编译好的 cubin
    3. 如果没有 → 编译 (AST → TTIR → TTGIR → LLVM → PTX → cubin)
    4. 加载 cubin 到 GPU driver
    5. 启动 kernel""")

    # ── 3. 编译并获取 IR (程序化) ─────────────────────────────
    print("─" * 70)
    print("  3. 程序化获取 IR (尝试多种方法)")
    print("─" * 70)

    # 方法 1: 通过运行 kernel + dump env vars (最可靠)
    print("\n  方法 1: 运行 kernel + TRITON_KERNEL_DUMP=1 (最可靠)")
    os.environ["TRITON_KERNEL_DUMP"] = "1"
    os.environ["TRITON_KERNEL_OVERRIDE"] = "1"

    N = 256
    x = torch.randn(N, device="cuda")
    y = torch.randn(N, device="cuda")
    out = torch.empty(N, device="cuda")
    api_test_kernel[(1,)](x, y, out, N, BLOCK=256)
    torch.cuda.synchronize()

    cache = Path.home() / ".triton" / "cache"
    if cache.exists():
        # 找最新修改的目录
        dirs = sorted(
            [d for d in cache.iterdir() if d.is_dir()],
            key=lambda p: p.stat().st_mtime, reverse=True
        )
        if dirs:
            latest = dirs[0]
            print(f"  最新 cache 目录: {latest.name}")
            for suffix, label in [(".ttir", "TTIR"), (".ttgir", "TTGIR"),
                                    (".ll", "LLVM"), (".ptx", "PTX")]:
                files = list(latest.glob(f"*{suffix}"))
                if files:
                    print(f"    {label:6s}: {files[0].name} ({files[0].stat().st_size} bytes)")

    # 方法 2: triton.compile (尝试)
    print("\n  方法 2: triton.compile() API (如果可用)")
    try:
        from triton.compiler import compile as triton_compile
        print("  ✅ triton.compiler.compile 可用")
        # 注意: 具体 API 参数取决于 Triton 版本
        # 这里只展示概念
        print("  使用方式 (取决于版本):")
        print("    from triton.compiler import compile")
        print("    result = compile(kernel_fn, signature, constants)")
        print("    # result 包含各阶段 IR")
    except (ImportError, AttributeError):
        print("  ⚠  triton.compiler.compile 不可用")
        print("  (Triton 3.x 中编译 API 仍在发展中)")

    # ── 4. Autotune 的 cache 机制 ────────────────────────────
    print("\n" + "─" * 70)
    print("  4. Autotune 的 cache 机制")
    print("─" * 70)
    print("""
  Triton 的 autotune cache 结构 (~/.triton/cache/):

  cache/
  ├── <hash_kernel1>/
  │   ├── <hash_config1>/
  │   │   ├── *.ttir      ← 这个 config 的 TTIR
  │   │   ├── *.ttgir     ← 这个 config 的 TTGIR
  │   │   ├── *.ptx       ← 这个 config 的 PTX
  │   │   └── *.cubin      ← 编译好的 GPU 二进制
  │   ├── <hash_config2>/
  │   │   └── ...
  │   └── autotune_cache.json  ← 记录了哪个 config 最快
  └── <hash_kernel2>/
      └── ...

  Cache 失效条件 (任何一个变化 → 重新编译):
    • kernel 源码 (包括任何微小的改动)
    • constexpr 参数的值
    • autotune config (num_warps, num_stages, BLOCK_SIZE)
    • GPU 架构 (SM version)
    • Triton 版本

  清空 cache:
    rm -rf ~/.triton/cache/
    → 下次运行全部重新编译 (第一次会很慢)""")

    # ── 5. Triton 内部: 从调用到执行 ──────────────────────
    print("─" * 70)
    print("  5. 完整调用链: kernel[grid](args...) 内部发生了什么")
    print("─" * 70)
    print("""
  kernel[grid](x, y, out, N, BLOCK_SIZE=256)

  1. JITFunction.__getitem__(grid)  → 返回一个 "launcher" 对象

  2. launcher.__call__(*args, **kwargs)
     → 收集参数类型: x=ptr(f32), y=ptr(f32), out=ptr(f32), N=i32

  3. 计算 specialization key:
     key = hash(
       kernel_source_code,
       constexpr_values={BLOCK_SIZE=256},
       gpu_arch="sm90",
       triton_version="3.6.0"
     )

  4. 检查 ~/.triton/cache/<key>/ 是否存在且有效
     √ 有 → 跳到步骤 7

  5. 编译 (没有 cache 时):
     a. parse Python AST → 构建 TTIR builder
     b. 运行 TTIR builder → 生成 TTIR (MLIR)
     c. 运行 compiler passes → TTIR → TTGIR → LLVM IR
     d. LLVM NVPTX backend → PTX
     e. ptxas → CUBIN
     f. 写入 cache

  6. 写入 cache → 写入 ~/.triton/cache/<key>/*.{ttir,ttgir,ll,ptx,cubin}

  7. 加载 CUBIN:
     cuModuleLoadData(cubin)      → CUDA module
     cuModuleGetFunction(module, "api_test_kernel") → kernel function

  8. 启动:
     cuLaunchKernel(
       function, grid, block,
       shared_mem_bytes, stream, args
     )

  9. kernel 在 GPU 上异步执行 → 你的代码继续运行
     (直到 torch.cuda.synchronize() 或需要结果时等待)""")

    print("\n📖 下一步: python phase4_compiler/13_custom_pass.py")
    print("   了解 MLIR Pass 的概念和 Python 绑定。\n")


if __name__ == "__main__":
    main()
