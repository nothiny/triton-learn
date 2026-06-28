"""
21_ptx_to_sass.py — PTX → SASS: GPU 机器码最终形态

学习目标:
  1. 理解 PTX 和 SASS 的本质区别
  2. 学会用 cuobjdump 反汇编 SASS
  3. 看到寄存器分配和指令调度在 SASS 中如何体现

运行: python phase4_compiler/21_ptx_to_sass.py

前提: 已完成 01-20。
     需要 CUDA toolkit (cuobjdump 或 nvdisasm)。
"""

import os
import sys
import subprocess
from pathlib import Path

os.environ["TRITON_KERNEL_DUMP"] = "1"
os.environ["TRITON_KERNEL_OVERRIDE"] = "1"

import torch
import triton
import triton.language as tl


# ══════════════════════════════════════════════════════════════════════
# 一个简单 kernel，用于分析 SASS
# ══════════════════════════════════════════════════════════════════════


@triton.jit
def kernel_for_sass(x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
    """最简单的 kernel，方便看 SASS。"""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


# ══════════════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════════════


def find_latest_cubin():
    cache = Path.home() / ".triton" / "cache"
    if not cache.exists():
        return None
    files = sorted(cache.rglob("*.cubin"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def find_latest_ptx():
    cache = Path.home() / ".triton" / "cache"
    if not cache.exists():
        return None
    files = sorted(cache.rglob("*.ptx"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def find_cuda_tool(tool_name):
    """查找 CUDA 工具 (cuobjdump 或 nvdisasm)。"""
    # 首先在 PATH 中找
    result = subprocess.run(["which", tool_name], capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()

    # 在常见的 CUDA 安装路径中找
    for cuda_home in [
        "/usr/local/cuda",
        os.environ.get("CUDA_HOME", ""),
        os.environ.get("CUDA_PATH", ""),
    ]:
        if cuda_home:
            path = os.path.join(cuda_home, "bin", tool_name)
            if os.path.isfile(path):
                return path
    return None


def try_disassemble(cubin_path):
    """尝试用 cuobjdump 或 nvdisasm 反汇编 cubin。"""
    # 试 cuobjdump
    cuobjdump = find_cuda_tool("cuobjdump")
    if cuobjdump:
        result = subprocess.run(
            [cuobjdump, "-sass", str(cubin_path)],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            return "cuobjdump", result.stdout
        # 如果 cuobjdump -sass 不可用，尝试 -ptx
        result = subprocess.run(
            [cuobjdump, "-ptx", str(cubin_path)],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            return "cuobjdump (PTX)", result.stdout

    # 试 nvdisasm
    nvdisasm = find_cuda_tool("nvdisasm")
    if nvdisasm:
        result = subprocess.run(
            [nvdisasm, "-ndf", "-c", str(cubin_path)],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            return "nvdisasm", result.stdout

    return None, None


# ══════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════


def main():
    print("=" * 70)
    print("  21 — PTX → SASS: GPU 机器码最终形态")
    print("=" * 70)

    # ── PTX vs SASS 概念 ──────────────────────────────────
    print("─" * 70)
    print("  1. PTX vs SASS: 有什么区别?")
    print("─" * 70)
    print("""
  ╔══════════════════════════════════════════════════════════════════╗
  ║  PTX = Portable/Intermediate  (可移植汇编)                      ║
  ║  SASS = Native/Machine Code   (原生机器码)                      ║
  ╚══════════════════════════════════════════════════════════════════╝

  编译链:
    Triton → TTIR → TTGIR → LLVM IR → PTX → SASS
                                          ↑      ↑
                                      ptxas 做的  这就是最终二进制!

  PTX (Parallel Thread Execution):
    • NVIDIA 定义的"虚拟 ISA" — 跨 GPU 架构兼容
    • 一个 PTX 文件可以在多代 GPU 上运行 (A100, H100, ...)
    • 其中 .reg 声明是"需求"，不是真正的物理寄存器
    • 包含符号信息 (.entry, .param)

  SASS (Streaming Assembler):
    • 特定 GPU 架构的原生机器码 — 每个架构有自己的 SASS
    • A100 (SM80) 和 H100 (SM90) 的 SASS 不同
    • 真正的物理寄存器编号 (R0, R1, R2, ...)
    • 指令已经被调度重排
    • 寄存器 spilling 在这里是"真实的" (溢出到 local memory)

  🔑 为什么看 SASS?
    PTX 是"需求"，SASS 是"现实"。
    例如:
      PTX: 声明 .reg .f32 %r<255>  (需要 255 个寄存器)
      SASS: 可能只用 128 个物理寄存器 (剩下被 spilling)
      → 只有看 SASS 才能确认到底 spill 了多少
""")

    # ── 运行 kernel 生成 cubin ────────────────────────────
    print("─" * 70)
    print("  2. 生成并找到 CUBIN")
    print("─" * 70)

    N = 256
    x = torch.randn(N, device="cuda")
    y = torch.randn(N, device="cuda")
    out = torch.empty(N, device="cuda")
    kernel_for_sass[(1,)](x, y, out, N, BLOCK=256)
    torch.cuda.synchronize()

    cubin = find_latest_cubin()
    if cubin:
        print(f"  CUBIN 文件: {cubin}")
        print(f"  大小: {cubin.stat().st_size} bytes")
    else:
        print("  ⚠ 未找到 .cubin 文件 (Triton 可能只在 GPU 上编译，不写入磁盘)")
        print("  尝试从 PTX 分析...")
        cubin = None

    # ── 反汇编 ────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  3. 反汇编 SASS")
    print("─" * 70)

    if cubin:
        tool_name, sass = try_disassemble(cubin)
        if tool_name and sass:
            print(f"  使用: {tool_name}")
            print(f"\n  SASS 代码 (前 50 行):")
            print(f"  {'─' * 60}")
            for line in sass.split("\n")[:50]:
                print(f"  {line}")
            if sass.count("\n") > 50:
                print(f"  ... (省略 {sass.count(chr(10)) - 50} 行)")
        else:
            print("  ⚠ cuobjdump / nvdisasm 不可用或反汇编失败")
            print("  请检查 CUDA toolkit 安装")
    else:
        print("  没有 CUBIN 文件，无法反汇编。")
        print("  跳过此步骤 (这不影响学习)。")

    # ── SASS 关键特征 ────────────────────────────────────
    print("\n" + "─" * 70)
    print("  4. SASS 中能看到什么 (PTX 中看不到的)")
    print("─" * 70)
    print("""
  PTX 是"虚拟的"，SASS 是"真实的"。以下信息只在 SASS 中可见:

  ❶ 真实的寄存器分配:
    PTX:  .reg .f32 %r<100>          ← 需要 100 个寄存器
    SASS: 使用 R0-R127               ← 实际用了 128 个
          或者: STL [R1+0x20], R128  ← spilling! (store to local memory)

  ❷ 指令调度 (Instruction Scheduling):
    PTX:  顺序排列的指令
    SASS: 指令被重排 (reorder) 以隐藏延迟
          两个连续的 ld.global 之间可能插入独立计算

  ❸ 指令延迟信息:
    PTX:  不包含延迟信息
    SASS: 可以通过 stall cycle 分析推断延迟

  ❹ 控制码 (Control Codes):
    SASS .code 段包含:
      • reuse flags: 哪些结果被复用
      • wait barriers: 哪些指令在等数据
      • yield hints: warp scheduler 在哪里可能切换

  ❺ 双发射 (Dual Issue):
    SASS 中可以看到哪些指令同时发射 (同 cycle)
    例如: FADD + FFMA 可以在同 cycle 执行

  ⚠ 大多数情况下你不需要看 SASS。
    但如果遇到:
      • PTX 显示没有 spill 但 ncu 显示大量 local memory 访问
      • 性能远低于 PTX 估算的理论值
      • 不同 GPU 架构上性能差异巨大
    → 那是时候看 SASS 了。
""")

    # ── 对比 PTX vs SASS (手动的) ─────────────────────────
    print("─" * 70)
    print("  5. PTX → SASS 对照示例 (NVIDIA 官方文档 + 推测)")
    print("─" * 70)
    print("""
  指令对照 (简化):
  ┌─────────────────────────────┬──────────────────────────────────┐
  │ PTX                          │ SASS (SM80/SM90)                  │
  ├─────────────────────────────┼──────────────────────────────────┤
  │ ld.global.ca.f32 %f1, [%rd] │ LDG.E.CA R2, [R4]                │
  │ st.global.f32 [%rd], %f1    │ STG.E [R4], R2                   │
  │ add.f32 %f3, %f1, %f2       │ FADD R6, R2, R4                  │
  │ fma.rn.f32 %f3, %f1, %f2, %f0 │ FFMA R6, R2, R4, R0            │
  │ mma.sync.aligned.m16n8k16   │ HMMA.16816.F32.F16.F16.F32 ...   │
  │ bar.sync 0                   │ BAR.SYNC 0x0                     │
  │ bra LABEL                    │ BRA LABEL                        │
  └─────────────────────────────┴──────────────────────────────────┘

  寄存器映射:
    PTX 中的 %r100  ≠  SASS 中的 R100
    PTX 寄存器是"虚拟寄存器"，SASS 寄存器是物理寄存器。
    映射由 ptxas 决定，可能完全不同。

  Spilling 检测:
    PTX: 没有 st.local 指令 → 但 SASS 中可能有!
    → 因为 ptxas 发现物理寄存器不够，即使 PTX 声明了足够的虚拟寄存器
    → 这就是看 SASS 的价值

  SASS 中的 Local Memory:
    STL [R1+offset], Rxx    ← spill (寄存器溢出到 stack)
    LDL Rxx, [R1+offset]    ← reload (从 stack 恢复到寄存器)
    R1 通常是 stack pointer

    grep "STL\|LDL" sass_output.txt → 找到 spill/reload
""")

    # ── 分析 PTX (fallback) ───────────────────────────────
    print("─" * 70)
    print("  6. 分析 PTX 的寄存器分配 (备选方案)")
    print("─" * 70)

    ptx = find_latest_ptx()
    if ptx:
        content = ptx.read_text()
        import re
        # 统计 .reg 声明
        reg_types = {}
        for m in re.finditer(r'\.reg\s+\.(\w+)\s+', content):
            t = m.group(1)
            reg_types[t] = reg_types.get(t, 0) + 1
        print(f"  PTX 寄存器声明:")
        total = 0
        for t, count in reg_types.items():
            equiv = count * (2 if t in ("b64", "f64") else 1)
            total += equiv
            print(f"    .{t}: {count} declarations (~{equiv} 32-bit)")
        print(f"    估算 32-bit 寄存器总数/线程: {total}")

        # 找 st.local (PTX 中的 spill 标志)
        has_spill = "st.local" in content or "ld.local" in content
        print(f"    PTX 中有 st.local/ld.local (spill): {has_spill}")

    # ── 总结 ──
    print("\n" + "─" * 70)
    print("  7. PTX → SASS 分析工作流")
    print("─" * 70)
    print("""
  1. 生成 CUBIN:
     TRITON_KERNEL_DUMP=1 python my_kernel.py
     → 在 ~/.triton/cache/... 找到 .cubin

  2. 反汇编:
     cuobjdump -sass <cubin_path> > kernel.sass
     或
     nvdisasm -ndf -c <cubin_path> > kernel.sass

  3. 分析 SASS:
     • 找物理寄存器使用: grep "R[0-9]" kernel.sass
     • 找 spill: grep "STL\|LDL" kernel.sass
     • 找 MMA: grep "HMMA\|IMMA" kernel.sass
     • 找 barrier: grep "BAR.SYNC" kernel.sass
     • 统计指令类型分布

  4. 对比 PTX vs SASS:
     • PTX 中多少虚拟寄存器 → SASS 中多少物理寄存器
     • PTX 中没声明 spill → SASS 中是否有 STL/LDL
     • MMA 指令是否和 PTX 中一致

  ⚠ 注意:
    • SASS 是架构特定的 (A100 ≠ H100)
    • ptxas 的优化策略可能变化 (CUDA 版本影响)
    • 大多数性能分析在 PTX 级别就足够了
    • 只有在极端性能调优时才需要看 SASS
""")

    print("\n🏁 Phase 4 进阶系列完成！")
    print("\n  你已学完 Phase 4 的全部内容:\n")
    print("  基础篇 (01-13):")
    print("    01-07: 逐层 IR 详解")
    print("    08-10: Pass 管线 + Pipelining + 寄存器")
    print("    11-13: 调试实战 + Compiler API + Custom Pass")
    print("\n  进阶篇 (14-21):")
    print("    14: AST → TTIR 内部机制")
    print("    15: 内存模型深度解析")
    print("    16: MMA/Tensor Core 深度")
    print("    17: 单操作 Lowering 全追踪")
    print("    18: Autotuner 内部机制")
    print("    19: 环境变量速查手册")
    print("    20: 源码导航")
    print("    21: PTX → SASS 最终机器码")
    print()


if __name__ == "__main__":
    main()
