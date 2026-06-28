"""
01_first_ir.py — 第一次接触 Triton 的 4 层中间表示

学习目标:
  1. 设置环境变量 dump 出所有编译阶段的 IR
  2. 找到并阅读 TTIR / TTGIR / LLVM IR / PTX 文件
  3. 建立"编译器管线"的心智模型

运行: python phase4_compiler/01_first_ir.py

前提: 你写过 Triton kernel（比如 phase1 的 vector add），但从未看过编译器的内部。
"""

import os
import sys
from pathlib import Path

# ── 第一步: 启用 IR dump ─────────────────────────────────────────────
# Triton 通过环境变量控制 IR dump。
# TRITON_KERNEL_DUMP=1 会在每次编译后把所有 IR 阶段写入 ~/.triton/cache/
os.environ["TRITON_KERNEL_DUMP"] = "1"
os.environ["TRITON_KERNEL_OVERRIDE"] = "1"  # 强制重新编译（跳过 JIT 缓存）

import torch
import triton
import triton.language as tl


# ══════════════════════════════════════════════════════════════════════
# 定义一个最简单的 kernel，够简单才能看清编译器的每一步
# ══════════════════════════════════════════════════════════════════════


@triton.jit
def vector_add(x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    功能: out = x + y （element-wise）

    编译器要做的事（你现在看不到，但之后每层 IR 会展示）:
      1. 把 BLOCK_SIZE 这个 Python 变量变成编译期常量
      2. 把 tl.arange(0, BLOCK_SIZE) 翻译成"生成 [0,1,2,...,BLOCK-1]"
      3. 把 tl.load/tl.store 翻译成 GPU 内存加载/存储指令
      4. 决定每个线程处理哪些元素（layout encoding）
      5. 最终生成 PTX 汇编
    """
    pid = tl.program_id(axis=0)                        # block 索引
    block_start = pid * BLOCK_SIZE                      # 这个 block 从哪里开始
    offsets = block_start + tl.arange(0, BLOCK_SIZE)    # 这个 block 处理的所有偏移
    mask = offsets < N                                   # 越界检查

    x = tl.load(x_ptr + offsets, mask=mask)              # 从 HBM 加载
    y = tl.load(y_ptr + offsets, mask=mask)
    z = x + y                                           # 逐元素加法
    tl.store(out_ptr + offsets, z, mask=mask)            # 写回 HBM


# ══════════════════════════════════════════════════════════════════════
# 辅助函数: 找到并展示编译产物
# ══════════════════════════════════════════════════════════════════════


def find_dumped_ir():
    """找到 Triton 刚刚 dump 的所有 IR 文件。"""
    cache = Path.home() / ".triton" / "cache"
    if not cache.exists():
        print("  ⚠  缓存目录不存在:", cache)
        return {}

    # 找最近修改的文件（按时间排序）
    all_files = sorted(cache.rglob("*"), key=lambda p: p.stat().st_mtime, reverse=True)

    # 按后缀分类
    ir_map = {}
    suffix_labels = {
        ".ttir":  "TTIR    (Triton IR)",
        ".ttgir": "TTGIR   (Triton GPU IR)",
        ".ll":    "LLVM IR",
        ".ptx":   "PTX     (GPU 汇编)",
    }
    for f in all_files:
        if f.suffix in suffix_labels:
            label = suffix_labels[f.suffix]
            if label not in ir_map:  # 只保留最新的
                ir_map[label] = f

    return ir_map


def print_section(title, content, max_lines=60):
    """带标题地打印内容，截断过长输出。"""
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}")
    lines = content.strip().split("\n")
    for line in lines[:max_lines]:
        print(f"  {line}")
    if len(lines) > max_lines:
        print(f"  ... (省略 {len(lines) - max_lines} 行)")


# ══════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════


def main():
    print("=" * 70)
    print("  01 — 第一次接触 Triton 的 4 层中间表示")
    print("=" * 70)

    # ── 步骤 1: 运行 kernel 触发编译 ──────────────────────────────
    print("\n📦 步骤 1: 运行 vector_add kernel 触发编译...")
    N = 1024
    BLOCK_SIZE = 256
    x = torch.randn(N, device="cuda")
    y = torch.randn(N, device="cuda")
    out = torch.empty(N, device="cuda")

    # grid: 向上取整 (N + BLOCK_SIZE - 1) // BLOCK_SIZE = ceil(N/BLOCK_SIZE)
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    vector_add[grid](x, y, out, N, BLOCK_SIZE=BLOCK_SIZE)
    torch.cuda.synchronize()

    # 验证正确性
    expected = x + y
    assert torch.allclose(out, expected), "Kernel 结果错误！"
    print("  ✅ kernel 执行成功，结果正确")

    # ── 步骤 2: 找到编译产物 ────────────────────────────────────
    print("\n🔍 步骤 2: 在 ~/.triton/cache/ 中查找编译产物...")
    ir_files = find_dumped_ir()

    if not ir_files:
        print("  ⚠  未找到 IR 文件。可能的原因:")
        print("     1. TRITON_KERNEL_DUMP 未生效（检查 Triton 版本）")
        print("     2. kernel 从缓存中加载（尝试 rm -rf ~/.triton/cache/）")
        print("\n  手动查找:")
        cache = Path.home() / ".triton" / "cache"
        if cache.exists():
            recent = sorted(cache.rglob("*.ptx"), key=lambda p: p.stat().st_mtime, reverse=True)
            if recent:
                print(f"     最新 PTX: {recent[0]}")
        return

    for label, filepath in ir_files.items():
        print(f"  📄 {label:30s} → {filepath.name}")

    # ── 步骤 3: 阅读每一层 IR ──────────────────────────────────
    print("\n" + "=" * 70)
    print("  步骤 3: 逐层阅读 IR — 注意每一层「多了什么」和「少了什么」")
    print("=" * 70)

    # 3.1 TTIR — 纯数学描述
    ttir_files = [f for label, f in ir_files.items() if "TTIR" in label]
    if ttir_files:
        content = ttir_files[0].read_text()
        print_section("1/4 — TTIR (Triton IR)", content, max_lines=60)
        print("""
  🔑 TTIR 的特点:
     • 只有"做什么运算"的信息（load, store, arith.addf）
     • 没有"哪个线程做"的信息 — 所有 tensor 都是完整的
     • 没有 GPU 硬件概念 — 你把这段 IR 给 CPU 编译器也能理解
     • 这是"最接近你 Python 代码"的 IR 层级
     • 关键 op: tt.load, tt.store, tt.arange, tt.program_id""")

    # 3.2 TTGIR — 加了线程分配
    ttgir_files = [f for label, f in ir_files.items() if "TTGIR" in label]
    if ttgir_files:
        content = ttgir_files[0].read_text()
        print_section("2/4 — TTGIR (Triton GPU IR)", content, max_lines=60)
        print("""
  🔑 TTGIR 的特点:
     • 每个 tensor 类型后面多了 #blocked<{...}> — 这就是 layout encoding!
     • layout encoding 回答了"哪个线程处理哪些数据元素"
     • 可能出现 ttg.convert_layout — 数据重新排列（潜在的性能代价）
     • 这是 Triton 编译器最独特的设计 — 传统编译器没有这一层""")

    # 3.3 LLVM IR — 通用底层
    ll_files = [f for label, f in ir_files.items() if "LLVM" in label]
    if ll_files:
        content = ll_files[0].read_text()
        print_section("3/4 — LLVM IR", content, max_lines=60)
        print("""
  🔑 LLVM IR 的特点:
     • 出现了具体的寄存器（%r1, %r2...）
     • 出现了显式的地址计算（getelementptr）
     • 出现了 threadIdx.x 的读取（nvvm.read.ptx.sreg.tid.x）
     • 但仍然是"通用"的 — LLVM 不知道这是给 H100 还是 A100 的
     • 寄存器分配在这里由 LLVM 完成（Triton 不管寄存器的具体分配）""")

    # 3.4 PTX — GPU 汇编
    ptx_files = [f for label, f in ir_files.items() if "PTX" in label]
    if ptx_files:
        content = ptx_files[0].read_text()
        print_section("4/4 — PTX (GPU 汇编)", content, max_lines=80)
        print("""
  🔑 PTX 的特点:
     • 这是真正在 GPU 上执行的指令！
     • ld.global.f32 = 从 HBM（显存）加载 32-bit 浮点数
     • st.global.f32 = 存储到 HBM
     • add.f32 = 浮点加法
     • .reg .f32 %r<N> 告诉你有多少个寄存器
     • 最终的 ptxas 汇编器还会把 PTX 变成 SASS（机器码）""")

    # ── 步骤 4: 总结 — 编译器管线全景 ──────────────────────
    print("\n" + "=" * 70)
    print("  步骤 4: 编译器管线全景图")
    print("=" * 70)
    print("""
  你的 Python 代码 (@triton.jit)
      │
      │  Triton 解析 Python AST（语法树）
      ▼
  ┌─────────────────────────────────┐
  │  TTIR (tt dialect, MLIR)        │  ← "做了什么运算？"
  │  纯数学描述，无 GPU 信息          │     load, store, addf, dot
  └──────────────┬──────────────────┘
                 │  ConvertTritonToTritonGPU  ← 最关键的 pass!
                 ▼
  ┌─────────────────────────────────┐
  │  TTGIR (ttg dialect, MLIR)      │  ← "哪个线程处理哪些数据？"
  │  每个 tensor 带 layout encoding   │     #blocked<...>, #mma<...>
  └──────────────┬──────────────────┘
                 │  ConvertTritonGPUToLLVM
                 ▼
  ┌─────────────────────────────────┐
  │  LLVM IR (NVPTX target)         │  ← "寄存器怎么分配？地址怎么算？"
  │  通用底层表示                     │     getelementptr, nvvm intrinsics
  └──────────────┬──────────────────┘
                 │  LLVM NVPTX backend
                 ▼
  ┌─────────────────────────────────┐
  │  PTX (NVIDIA GPU 汇编)          │  ← 真正在 GPU 上执行的指令
  │  ld.global, add.f32, st.global  │
  └──────────────┬──────────────────┘
                 │  ptxas (NVIDIA 汇编器)
                 ▼
             SASS / CUBIN
            (GPU 机器码 010101...)

  关键心智模型:
    每一层 IR 都在做两件事:
      1. 丢掉一些上层信息（Python 语义 → 纯数学 → ...）
      2. 加入一些下层信息（...→ 线程分配 → 寄存器 → 具体指令）

    理解这个过程 = 理解编译器如何"看到"你的代码。""")

    print("\n📖 下一步: python phase4_compiler/02_ttir_language.py")
    print("   深入理解 TTIR 的每个 op，看看你的 Python 代码变成了什么样的数学描述。\n")


if __name__ == "__main__":
    main()
