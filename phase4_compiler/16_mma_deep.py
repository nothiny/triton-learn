"""
16_mma_deep.py — Tensor Core MMA 内部机制深度解析

学习目标:
  1. 理解 MMA (Matrix Multiply-Accumulate) 的硬件机制
  2. 知道 Ampere (SM80) vs Hopper (SM90) 的 MMA 差异
  3. 看懂 MMA layout encoding 的每个参数
  4. 理解为什么 tl.dot 对 dtype 和 shape 有严格要求

运行: python phase4_compiler/16_mma_deep.py

前提: 已完成 01-15。
"""

import os
from pathlib import Path

os.environ["TRITON_KERNEL_DUMP"] = "1"
os.environ["TRITON_KERNEL_OVERRIDE"] = "1"

import torch
import triton
import triton.language as tl


# ══════════════════════════════════════════════════════════════════════
# MMA 相关的 kernel
# ══════════════════════════════════════════════════════════════════════


@triton.jit
def mma_demo_fp16(A, B, C,
                   M, N, K,
                   BLOCK_M: tl.constexpr,
                   BLOCK_N: tl.constexpr,
                   BLOCK_K: tl.constexpr):
    """
    标准 fp16 MMA — 观察 Ampere m16n8k16 和 Hopper m16n8k32。
    """
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
    tl.store(C + rm[:, None] * N + rn[None, :], acc)


@triton.jit
def mma_demo_fp32(A, B, C,
                   M, N, K,
                   BLOCK_M: tl.constexpr,
                   BLOCK_N: tl.constexpr,
                   BLOCK_K: tl.constexpr):
    """
    fp32 MMA — 仅在 Ampere 上支持 (m16n8k8, 较小的 tile)。
    Hopper 不支持 fp32 MMA (会退化为 elementwise FMA)。
    """
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
        acc += tl.dot(a, b)          # fp32 × fp32 → m16n8k8 (Ampere)
    tl.store(C + rm[:, None] * N + rn[None, :], acc)


# ══════════════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════════════


def find_latest_ir(suffix):
    cache = Path.home() / ".triton" / "cache"
    if not cache.exists():
        return None
    files = sorted(cache.rglob(f"*.{suffix}"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def extract_mma_info(ptx_source):
    """从 PTX 中提取 MMA 相关信息。"""
    import re
    mma_lines = [l.strip() for l in ptx_source.split("\n") if "mma.sync" in l]
    info = {
        "num_mma_ops": len(mma_lines),
        "mma_shapes": set(),
    }
    for line in mma_lines:
        m = re.search(r'm\d+n\d+k\d+', line)
        if m:
            info["mma_shapes"].add(m.group())
    return info, mma_lines


# ══════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════


def main():
    print("=" * 70)
    print("  16 — Tensor Core MMA 内部机制深度解析")
    print("=" * 70)

    # ── MMA 基础概念 ──────────────────────────────────────
    print("─" * 70)
    print("  1. Tensor Core MMA 是什么?")
    print("─" * 70)
    print("""
  ╔══════════════════════════════════════════════════════════════════╗
  ║  Tensor Core = GPU 上的专用矩阵乘法硬件单元                      ║
  ║  一个 MMA 指令 = D = A × B + C  (全部在一个 cycle 中完成)       ║
  ╚══════════════════════════════════════════════════════════════════╝

  普通 ALU (CUDA Core):
    • 标量指令: add.f32, mul.f32, fma.f32
    • 一个 cycle 处理 1 个标量
    • 吞吐: ~128 FMA/cycle/SM (A100)

  Tensor Core:
    • 矩阵指令: mma.sync.aligned.m16n8k16
    • 一个 cycle 处理 16×8×16 = 2048 个乘加 = 4096 FLOPs
    • 吞吐: ~256 FMA/cycle/SM (A100) → 2x 于 CUDA Core

  为什么快?
    • 不是"1 个乘法 + 1 个加法"分开做
    • 而是一次完成一个 16×8×16 的矩阵乘加
    • 数据在 warp 内的 32 个线程之间通过 warp-level 寄存器共享
""")

    # ── MMA 指令形状 ──────────────────────────────────────
    print("─" * 70)
    print("  2. MMA 指令形状 (Instr Shape)")
    print("─" * 70)
    print("""
  不同架构的 MMA 指令支持不同的形状:

  ┌──────────────┬──────────────────────────────────────────────────┐
  │ 架构           │ 支持的 MMA 形状                                   │
  ├──────────────┼──────────────────────────────────────────────────┤
  │ A100 (SM80)  │ fp16: m16n8k16, m16n8k8                         │
  │ Ampere       │ bf16: m16n8k16, m16n8k8                         │
  │              │ tf32: m16n8k8                                     │
  │              │ fp64: m8n4k4 (吞吐低)                             │
  │              │ fp32: m16n8k8 (Triton 2.1+, 支持有限)             │
  ├──────────────┼──────────────────────────────────────────────────┤
  │ H100 (SM90)  │ fp16: m16n8k32, m16n8k16                        │
  │ Hopper       │ bf16: m16n8k32                                    │
  │              │ fp8:  m16n8k64 (e4m3/e5m2, 吞吐翻倍!)            │
  │              │ tf32: m16n8k16                                    │
  │              │ fp32: ❌ 不支持 MMA! (只支持 CUDA Core FMA)       │
  └──────────────┴──────────────────────────────────────────────────┘

  🔑 关键变化 (A100 → H100):
    • K 维度从 16 翻倍到 32 (m16n8k16 → m16n8k32)
    • 更大的 K tile → 更少的迭代 → 更高的效率
    • fp8 支持 (m16n8k64) → 吞吐再翻倍
    • fp32 MMA 被移除 → fp32 代码在 H100 上不能用 Tensor Core!

  Triton 自动选择:
    • tl.dot(a, b) 的 dtype 和 BLOCK_K 决定 MMA 形状
    • 编译器选择能覆盖 tile 的最小 MMA 形状
    • 如果找不到合适的 MMA 形状 → 退化为 elementwise FMA → 慢!
""")

    # ── 运行 fp16 MMA ─────────────────────────────────────
    print("─" * 70)
    print("  3. 观察 fp16 MMA 的 PTX 产出")
    print("─" * 70)

    M, N, K = 128, 128, 256
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float32)
    mma_demo_fp16[(1, 1)](A, B, C, M, N, K, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32)
    torch.cuda.synchronize()

    ptx_fp16 = find_latest_ir("ptx")
    if ptx_fp16:
        content = ptx_fp16.read_text()
        info, mma_lines = extract_mma_info(content)
        print(f"    MMA 操作数: {info['num_mma_ops']}")
        print(f"    MMA 形状: {info['mma_shapes']}")
        print(f"    示例指令:")
        for line in mma_lines[:5]:
            print(f"      {line[:120]}")

    # ── 运行 fp32 MMA ─────────────────────────────────────
    print("\n" + "─" * 70)
    print("  4. 观察 fp32 'MMA' (可能退化!)")
    print("─" * 70)

    M2, N2, K2 = 128, 128, 128  # smaller K to match m16n8k8
    A32 = torch.randn(M2, K2, device="cuda", dtype=torch.float32)
    B32 = torch.randn(K2, N2, device="cuda", dtype=torch.float32)
    C32 = torch.empty(M2, N2, device="cuda", dtype=torch.float32)

    fp32_error = None
    try:
        mma_demo_fp32[(1, 1)](A32, B32, C32, M2, N2, K2,
                                BLOCK_M=32, BLOCK_N=32, BLOCK_K=16)
        torch.cuda.synchronize()
    except Exception as e:
        fp32_error = str(e)

    ptx_fp32 = find_latest_ir("ptx")
    if ptx_fp32:
        content = ptx_fp32.read_text()
        has_mma = "mma.sync" in content
        has_fma = "fma.rn.f32" in content
        print(f"    有 mma.sync: {has_mma}")
        print(f"    有 fma.rn.f32: {has_fma}")
        if has_mma:
            info, _ = extract_mma_info(content)
            print(f"    MMA 形状: {info['mma_shapes']} (m16n8k8)")
        else:
            print(f"    ❌ 没有 MMA 指令! tl.dot 退化为 elementwise FMA")
            print(f"    这意味着性能损失 3-5x")
    elif fp32_error:
        print(f"    ⚠ fp32 MMA kernel 运行失败 (预期可能): {fp32_error[:150]}")
        print(f"    这正说明 fp32 MMA 在当前环境不受支持!")

    # ── MMA Layout Encoding 详解 ──────────────────────────
    print("\n" + "─" * 70)
    print("  5. MMA 相关的 Layout Encoding")
    print("─" * 70)
    print("""
  回忆: tl.dot 涉及三种 layout:

  1. 输入 A 的 layout: DotOperandEncodingAttr{opIdx=0, parent=...}
     • opIdx=0 → A 操作数
     • 数据按 K 维 innermost 排列 (每 warp 的寄存器中)

  2. 输入 B 的 layout: DotOperandEncodingAttr{opIdx=1, parent=...}
     • opIdx=1 → B 操作数
     • 同样 K 维 innermost

  3. 输出 C 的 layout: MmaEncodingAttr{versionMajor=2, versionMinor=0, instrShape=[16,8,16]}
     • 输出在 Tensor Core 的 warp 级矩阵布局中
     • instrShape 告诉编译器用哪个 MMA 指令

  数据流:
    Load (global → register, #blocked)
      → (如果需要) convert_layout → #dot_op
      → tl.dot: A(#dot_op) × B(#dot_op) → C(#mma)
      → (如果需要) convert_layout → #blocked
      → Store (register → global)

  🔑 关键: 如果 convert_layout 能消除 (TritonGPURemoveLayoutConversions):
    • Load 直接以 #dot_op layout 加载 → 不需要转换 → 零开销
    • 这是最高效的情况

  如果不能消除:
    • Load (#blocked) → convert → #dot_op → dot → 多一次 shared memory round-trip
""")

    # ── Warp 级 MMA 数据分布 ──────────────────────────────
    print("─" * 70)
    print("  6. Warp 级 MMA 数据分布 (m16n8k16 为例)")
    print("─" * 70)
    print("""
  一个 warp = 32 个线程。
  一个 mma.sync.aligned.m16n8k16 指令:
    • A: 16×16 matrix (fp16)  → 由 32 个线程的寄存器集体持有
    • B: 16×8  matrix (fp16)  → 由 32 个线程的寄存器集体持有
    • C/D: 16×8 matrix (fp32) → 由 32 个线程的寄存器集体持有

  数据分布 (简化):
    ┌─────────────────────────────────────┐
    │  Warp 的 32 个线程:                  │
    │                                      │
    │  Thread 0:  A[0:4, 0:2]  ← 4×2     │
    │  Thread 1:  A[4:8, 0:2]             │
    │  Thread 2:  A[8:12, 0:2]            │
    │  ...                                 │
    │  Thread 7:  A[28:32, 0:2]           │
    │  Thread 8:  A[0:4, 2:4]             │
    │  ...                                 │
    │                                      │
    │  每个线程持有 A 的一部分 (称为"fragment")│
    │  MMA 指令内部在线程间自动做数据交换     │
    └─────────────────────────────────────┘

  Triton 的 MmaEncodingAttr 管理这个分布。
  你不需要手动管理它 — 编译器为你处理。

  但理解这个分布有助于理解:
    • 为什么 BLOCK_M 必须是 16 的倍数 (match MMA M dim)
    • 为什么 BLOCK_N 必须是 8 的倍数  (match MMA N dim)
    • 为什么 BLOCK_K 必须是 16/32 的倍数 (match MMA K dim)
""")

    # ── 总结: tl.dot 的约束 ──────────────────────────────
    print("─" * 70)
    print("  7. tl.dot 的约束速查")
    print("─" * 70)
    print("""
  ┌──────────────────┬──────────────────────────────────────────────┐
  │ 约束              │ 原因                                          │
  ├──────────────────┼──────────────────────────────────────────────┤
  │ dtype=fp16/bf16  │ Tensor Core 原生支持这些格式                    │
  │ dtype=fp32       │ Ampere: m16n8k8 (小), Hopper: ❌ 不支持       │
  │ M 是 16 的倍数    │ MMA M 维度 = 16                               │
  │ N 是 8 的倍数     │ MMA N 维度 = 8                                │
  │ K 是 16 的倍数    │ MMA K 维度 = 16 (Ampere) 或 32 (Hopper)      │
  │ BLOCK_K ≥ 16     │ 至少一个完整的 MMA K tile                     │
  │ 累加器用 fp32    │ 精度! MMA 输出 fp32 → acc 应该是 fp32         │
  └──────────────────┴──────────────────────────────────────────────┘

  常见错误:
    ❌ K = 100 (不是 16 的倍数) → MMA 可能无法触发
    ❌ BLOCK_M = 60 (不是 16 的倍数) → 编译器可能退化
    ❌ dtype=fp32 on H100 → 没有 MMA, 只能用 FMA
    ✅ K = 128, BLOCK_M = 64, dtype=fp16 → 完美匹配
""")

    print("\n📖 下一步: python phase4_compiler/17_lowering_trace.py")
    print("   完整追踪一个 tl.dot 操作在各层 IR 中的 lowering 过程。\n")


if __name__ == "__main__":
    main()
