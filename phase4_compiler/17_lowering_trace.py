"""
17_lowering_trace.py — 单操作 Lowering 全流程追踪

学习目标:
  1. 追踪一个 tl.dot 从 Python → TTIR → TTGIR → LLVM → PTX 的完整过程
  2. 看到每个 pass 具体做了什么变换
  3. 建立"任何 op 都可以这样追踪"的通用方法论

运行: python phase4_compiler/17_lowering_trace.py

前提: 已完成 01-16。
"""

import os
from pathlib import Path

os.environ["TRITON_KERNEL_DUMP"] = "1"
os.environ["TRITON_KERNEL_OVERRIDE"] = "1"

import torch
import triton
import triton.language as tl


# ══════════════════════════════════════════════════════════════════════
# 追踪用的 kernel — 足够简单才能看清每个 pass 的变换
# ══════════════════════════════════════════════════════════════════════


@triton.jit
def trace_dot(A, B, C,
              M, N, K,
              BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr,
              BLOCK_K: tl.constexpr):
    """
    最简单的 matmul，只有一次 load→dot→store 迭代。
    没有循环，没有 pipelining，方便追踪。
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)    # [BLOCK_M]
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)    # [BLOCK_N]
    rk = tl.arange(0, BLOCK_K)                       # [BLOCK_K]

    # 指针偏移
    A_ptr = A + rm[:, None] * K + rk[None, :]       # [BLOCK_M, BLOCK_K]
    B_ptr = B + rk[:, None] * N + rn[None, :]       # [BLOCK_K, BLOCK_N]

    # Step 1: Load (Python → TTIR: tt.load)
    a = tl.load(A_ptr)
    b = tl.load(B_ptr)

    # Step 2: Dot (TTIR: tt.dot → TTGIR: 带 MMA layout)
    acc = tl.dot(a, b)

    # Step 3: Store (TTIR: tt.store)
    C_ptr = C + rm[:, None] * N + rn[None, :]       # [BLOCK_M, BLOCK_N]
    tl.store(C_ptr, acc)


# ══════════════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════════════


def find_latest_ir(suffix):
    cache = Path.home() / ".triton" / "cache"
    if not cache.exists():
        return None
    files = sorted(cache.rglob(f"*.{suffix}"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def grep_lines(source, keywords, context=0):
    """在 source 中找包含任意 keyword 的行。"""
    lines = source.split("\n")
    result = []
    for i, line in enumerate(lines):
        for kw in keywords:
            if kw in line:
                if context > 0:
                    start = max(0, i - context)
                    end = min(len(lines), i + context + 1)
                    result.extend(lines[start:end])
                else:
                    result.append(line.strip())
                break
    return result


# ══════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════


def main():
    print("=" * 70)
    print("  17 — 单操作 Lowering 全流程追踪")
    print("=" * 70)

    print("""
  ╔══════════════════════════════════════════════════════════════════╗
  ║  追踪对象: tl.dot(a, b)  where a=[32,16]fp16, b=[16,32]fp16   ║
  ║  追踪过程: Python → TTIR → TTGIR → LLVM → PTX                  ║
  ╚══════════════════════════════════════════════════════════════════╝

  我们将看到:
    Stage 1 (Python):  你在写什么
    Stage 2 (TTIR):    纯数学描述 — tt.dot
    Stage 3 (TTGIR):  加了 MMA layout — #dot_op + #mma
    Stage 4 (LLVM):    NVVM MMA intrinsic
    Stage 5 (PTX):     mma.sync 指令
""")

    # ── 运行 kernel ────────────────────────────────────────
    M, N, K = 32, 32, 16
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float32)

    trace_dot[(1, 1)](A, B, C, M, N, K, BLOCK_M=32, BLOCK_N=32, BLOCK_K=16)
    torch.cuda.synchronize()
    print("  ✅ Kernel 编译并运行成功\n")

    # ── Stage 1: Python 源码 ──────────────────────────────
    print("=" * 70)
    print("  Stage 1: Python 源码")
    print("=" * 70)
    print("""
    a = tl.load(A_ptr)     # 加载 (32, 16) fp16 tile
    b = tl.load(B_ptr)     # 加载 (16, 32) fp16 tile
    acc = tl.dot(a, b)     # 矩阵乘: (32,16) × (16,32) → (32,32) fp32

  编译器看到:
    • tl.load() 两次
    • tl.dot()  一次 — 这是"特殊操作"，会触发 MMA lowering
    • 输入 dtype = fp16, 输出 dtype = fp32 (累加器)
""")

    # ── Stage 2: TTIR ─────────────────────────────────────
    print("=" * 70)
    print("  Stage 2: TTIR (纯数学描述)")
    print("=" * 70)

    ttir = find_latest_ir("ttir")
    if ttir:
        content = ttir.read_text()
        # 找关键 op
        key_lines = grep_lines(content, ["tt.dot", "tt.load", "tt.store"])
        print(f"  文件: {ttir.name}")
        print(f"  关键 op:")
        for line in key_lines[:15]:
            print(f"    {line}")

    print("""
  🔑 TTIR 中的 tt.dot:
    • "这里需要一个矩阵乘法"
    • 没有指定用什么硬件来做 (Tensor Core 还是 CUDA Core)
    • 没有指定 MMA 的形状 (16×8×16 还是 8×8×16)
    • 这些决策在后面的 pass 中做出
""")

    # ── Stage 3: TTGIR ────────────────────────────────────
    print("=" * 70)
    print("  Stage 3: TTGIR (带 MMA layout)")
    print("=" * 70)

    ttgir = find_latest_ir("ttgir")
    if ttgir:
        content = ttgir.read_text()

        # 找 MMA 相关的 layout
        layout_lines = grep_lines(content, ["#mma", "#dot_op", "#blocked", "convert_layout"])
        print(f"  文件: {ttgir.name}")
        print(f"  Layout 信息:")
        for line in layout_lines[:10]:
            print(f"    {line[:150]}")

        # 检查 convert_layout
        n_convert = content.count("convert_layout")
        print(f"\n  convert_layout 出现次数: {n_convert}")

    print("""
  🔑 TTGIR 中的变化:
    After ConvertTritonToTritonGPU:
      • tt.dot 的输入变成了 #dot_op<{opIdx=0, ...}> 和 #dot_op<{opIdx=1, ...}>
      • tt.dot 的输出变成了 #mma<{versionMajor=2, instrShape=[16,8,16]}>

    After TritonGPUAccelerateMatmul:
      • tt.dot 被替换为 MMA 操作序列
      • 选择具体的 MMA 形状 (根据 GPU 架构和 dtype)
""")

    # ── Stage 4: LLVM IR ─────────────────────────────────
    print("=" * 70)
    print("  Stage 4: LLVM IR (NVVM MMA intrinsic)")
    print("=" * 70)

    ll = find_latest_ir("ll")
    if ll:
        content = ll.read_text()
        # 找 MMA intrinsic
        mma_lines = grep_lines(content, ["mma", "nvvm.mma", "wgmma"])
        print(f"  文件: {ll.name}")
        if mma_lines:
            print(f"  MMA 相关的 LLVM intrinsic:")
            for line in mma_lines[:5]:
                print(f"    {line[:150]}")
        else:
            print(f"  (MMA intrinsic 可能被展开为更底层的操作)")

    print("""
  🔑 LLVM IR 中的变化:
    After ConvertTritonGPUToLLVM:
      • Layout encoding 展开为显式的线程索引计算
      • tt.dot 的 MMA layout → NVVM MMA intrinsic:
        @llvm.nvvm.mma.m16n8k16.row.col.f32.f16.f16.f32
      • 寄存器开始出现
""")

    # ── Stage 5: PTX ──────────────────────────────────────
    print("=" * 70)
    print("  Stage 5: PTX (最终 GPU 指令)")
    print("=" * 70)

    ptx = find_latest_ir("ptx")
    if ptx:
        content = ptx.read_text()
        mma_lines = grep_lines(content, ["mma.sync", "ld.global", "st.global"])
        print(f"  文件: {ptx.name}")
        print(f"  关键指令:")
        for line in mma_lines[:10]:
            print(f"    {line[:150]}")

        # 检查是否有 shared memory 参与
        has_shared = "ld.shared" in content or "st.shared" in content
        print(f"\n  使用 shared memory: {has_shared}")
        if has_shared:
            n_shared = content.count("ld.shared") + content.count("st.shared")
            print(f"    shared memory 指令数: {n_shared}")
            print(f"    (如果有 convert_layout 或 wgmma, 可能会用到 shared memory)")

    # ── 完整的 lowering 流程图 ────────────────────────────
    print("\n" + "=" * 70)
    print("  完整 Lowering 流程图: tl.dot(a, b)")
    print("=" * 70)
    print("""
  Python:
    acc = tl.dot(a, b)         # a: (32,16)fp16, b: (16,32)fp16
         │
    ┌────▼────────────────────────────────────────────┐
    │ TTIR (tt dialect):                              │
    │   %acc = tt.dot %a, %b                         │
    │   : tensor<32x16xf16> × tensor<16x32xf16>     │
    │   → tensor<32x32xf32>                          │
    │                                                 │
    │   • 纯数学描述                                  │
    │   • 无 GPU 信息                                 │
    └────┬────────────────────────────────────────────┘
         │ ConvertTritonToTritonGPU
    ┌────▼────────────────────────────────────────────┐
    │ TTGIR (ttg dialect):                            │
    │   %a: tensor<32x16xf16,                         │
    │             #dot_op<{opIdx=0, ...}>>            │
    │   %b: tensor<16x32xf16,                         │
    │             #dot_op<{opIdx=1, ...}>>            │
    │   %acc: tensor<32x32xf32,                       │
    │               #mma<{instrShape=[16,8,16]}>>     │
    │                                                 │
    │   • 加了 layout encoding                        │
    │   • #dot_op → MMA 输入布局                      │
    │   • #mma → MMA 输出布局                         │
    └────┬────────────────────────────────────────────┘
         │ TritonGPUAccelerateMatmul
         │ (tt.dot → MMA op sequence)
    ┌────▼────────────────────────────────────────────┐
    │ TTGIR (after AccelerateMatmul):                 │
    │   %acc = ttg.mma %a, %b {                      │
    │     versionMajor=2, versionMinor=0,             │
    │     instrShape=[16,8,16]                        │
    │   }                                             │
    │                                                 │
    │   • tt.dot 被替换为 ttg.mma                     │
    │   • 选择了 m16n8k16 (如果 K=16)                  │
    └────┬────────────────────────────────────────────┘
         │ ConvertTritonGPUToLLVM
         │ (展开 layout → 显式寄存器)
    ┌────▼────────────────────────────────────────────┐
    │ LLVM IR:                                        │
    │   %mma = call @llvm.nvvm.mma.m16n8k16           │
    │          .row.col.f32.f16.f16.f32(...)         │
    │                                                 │
    │   • NVVM MMA intrinsic                         │
    │   • 操作数在寄存器中                              │
    └────┬────────────────────────────────────────────┘
         │ NVPTX CodeGen
    ┌────▼────────────────────────────────────────────┐
    │ PTX:                                            │
    │   mma.sync.aligned.m16n8k16.row.col.f32         │
    │       .f16.f16.f32 {%f1, %f2, ...},            │
    │       {%f3, %f4, ...}, {%f5, %f6, ...},        │
    │       {%f5, %f6, ...};                         │
    │                                                 │
    │   • 真正的 Tensor Core 指令!                    │
    │   • 每个 cycle 完成 16×8×16 = 4096 FLOPs      │
    └─────────────────────────────────────────────────┘

  🔑 每个阶段的核心决策:
    TTIR → TTGIR:          选择 layout (#dot_op + #mma)
    AccelerateMatmul:      选择 MMA 形状 (m16n8k16 vs m16n8k32)
    ConvertGPUToLLVM:      展开为 NVVM intrinsic
    NVPTX CodeGen:         寄存器分配 + 生成 mma.sync 指令
""")

    # ── 通用追踪方法论 ─────────────────────────────────────
    print("─" * 70)
    print("  通用追踪方法论: 如何追踪任意 op")
    print("─" * 70)
    print("""
  1. 写一个最简单的 kernel (只有你想追踪的那个 op)
  2. 运行: TRITON_KERNEL_DUMP=1 python my_kernel.py
  3. 在 ~/.triton/cache/ 找到 .ttir, .ttgir, .ll, .ptx
  4. 逐级阅读:
     • TTIR: 确认 op 被正确识别 (如 tt.dot 存在)
     • TTGIR: 看 layout encoding 和 convert_layout
     • LLVM: 看 NVVM intrinsic 和寄存器使用
     • PTX: 看最终指令 (mma.sync / fma / ...)

  5. 对比分析:
     • 对比不同 dtype (fp16 vs fp32) 的 PTX
     • 对比不同 BLOCK 大小的 TTGIR layout
     • 对比不同 GPU 架构的 MMA 形状

  这个方法适用于: tl.dot, tl.sum, tl.max, tl.load (mask),
                  任何你想深入理解的 Triton op。
""")

    print("\n📖 下一步: python phase4_compiler/18_autotuner.py")
    print("   深入 Triton 的 autotuner 内部机制。\n")


if __name__ == "__main__":
    main()
