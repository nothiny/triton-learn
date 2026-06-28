"""
07_ptx_assembly.py — PTX 汇编精读：GPU 真正执行的指令

学习目标:
  1. 读懂 PTX 的关键指令类别
  2. 建立 "PTX 指令 → GPU 硬件行为" 的映射
  3. 从 PTX 中判断寄存器压力、内存访问模式、是否用了 Tensor Core

运行: python phase4_compiler/07_ptx_assembly.py

前提: 已运行 06，理解 LLVM IR 的基本结构。
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
# 几个 kernel 用于生成不同类型的 PTX
# ══════════════════════════════════════════════════════════════════════


@triton.jit
def kernel_vector_ops(x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
    """
    包含多种算术操作: add, mul, fma, max, div
    用于观察 PTX 中的算术指令。
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    z = x * y + x                 # FMA
    w = tl.maximum(z, 0.0)        # max
    v = w / (w + 1.0)             # div + add
    tl.store(out_ptr + offs, v, mask=mask)


@triton.jit
def kernel_matmul_ptx(A, B, C,
                       M, N, K,
                       BLOCK_M: tl.constexpr,
                       BLOCK_N: tl.constexpr,
                       BLOCK_K: tl.constexpr):
    """
    简单的 matmul tile，用于观察 PTX 中的 mma.sync 指令。
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


# ══════════════════════════════════════════════════════════════════════
# 辅助函数: PTX 注释器
# ══════════════════════════════════════════════════════════════════════


def find_latest_ptx():
    cache = Path.home() / ".triton" / "cache"
    if not cache.exists():
        return None
    files = sorted(cache.rglob("*.ptx"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def annotate_ptx(ptx_source: str) -> str:
    """
    给 PTX 源代码加上中文注释，标注每条指令的含义和延迟。
    """
    annotations = {
        # ── 寄存器声明 ──
        r"^(\s*)\.reg\s+\.f32\s+(.*)": r"\1.reg .f32 \2  // [REG] float 寄存器声明",
        r"^(\s*)\.reg\s+\.b32\s+(.*)": r"\1.reg .b32 \2  // [REG] 32-bit 通用寄存器",
        r"^(\s*)\.reg\s+\.b64\s+(.*)": r"\1.reg .b64 \2  // [REG] 64-bit 地址寄存器",
        r"^(\s*)\.reg\s+\.pred\s+(.*)": r"\1.reg .pred \2  // [REG] 谓词 (predicate) 寄存器",

        # ── 共享内存 ──
        r"^(\s*)\.shared\s+(.*)": r"\1.shared \2  // [SHARED] 共享内存分配",

        # ── 全局内存访问 ──
        r"(ld\.global[^;]*)": r"\1  // [LD.GLOBAL] HBM 加载 (~300-800 周期，非常慢)",
        r"(st\.global[^;]*)": r"\1  // [ST.GLOBAL] HBM 存储 (~300-800 周期)",

        # ── 共享内存访问 ──
        r"(ld\.shared[^;]*)": r"\1  // [LD.SHARED] 共享内存加载 (~20-30 周期)",
        r"(st\.shared[^;]*)": r"\1  // [ST.SHARED] 共享内存存储 (~20-30 周期)",

        # ── 算术指令 ──
        r"(add\.f32[^;]*)": r"\1  // [ALU] 浮点加法",
        r"(mul\.f32[^;]*)": r"\1  // [ALU] 浮点乘法",
        r"(fma\.rn\.f32[^;]*)": r"\1  // [ALU] FMA (融合乘加, 1 周期)",
        r"(max\.f32[^;]*)": r"\1  // [ALU] 浮点 max",
        r"(div\.(?:approx\.)?f32[^;]*)": r"\1  // [ALU] 浮点除法 (较慢)",

        # ── Tensor Core ──
        r"(mma\.sync[^;]*)": r"\1  // [MMA] **Tensor Core** warp 级矩阵乘加!",
        r"(ld\.matrix[^;]*)": r"\1  // [MMA.LOAD] Tensor Core 矩阵加载",
        r"(st\.matrix[^;]*)": r"\1  // [MMA.STORE] Tensor Core 矩阵存储",

        # ── 同步 ──
        r"(bar\.sync[^;]*)": r"\1  // [BARRIER] block 级同步 (所有线程等齐)",
        r"(bar\.warp\.sync[^;]*)": r"\1  // [WARP.SYNC] warp 级同步",
        r"(membar\.cta[^;]*)": r"\1  // [MEMBAR] CTA 级内存屏障",
        r"(membar\.gl[^;]*)": r"\1  // [MEMBAR] 全局内存屏障",

        # ── 控制流 ──
        r"(@%p\d+\s+bra[^;]*)": r"\1  // [BRANCH] 条件分支",
        r"(ret[^;]*)": r"\1  // [RET] 函数返回",

        # ── 特殊寄存器 ──
        r"(%ctaid\.x)": r"\1  // blockIdx.x",
        r"(%tid\.x)": r"\1  // threadIdx.x",
        r"(%ntid\.x)": r"\1  // blockDim.x",
    }

    lines = ptx_source.split("\n")
    annotated = []
    for line in lines:
        stripped = line.strip()
        matched = False
        for pattern, replacement in annotations.items():
            if re.search(pattern, stripped):
                # 做一个简单的替换：在整个行后面加注释
                new_line = re.sub(pattern, replacement, stripped, count=1)
                annotated.append(new_line)
                matched = True
                break
        if not matched:
            annotated.append(stripped)
    return "\n".join(annotated)


def count_registers(ptx_source: str) -> dict:
    """统计 PTX 中声明的寄存器数量（每线程）。"""
    f32_regs = len(re.findall(r"\.reg\s+\.f32\s+", ptx_source))
    b32_regs = len(re.findall(r"\.reg\s+\.b32\s+", ptx_source))
    b64_regs = len(re.findall(r"\.reg\s+\.b64\s+", ptx_source))
    pred_regs = len(re.findall(r"\.reg\s+\.pred\s+", ptx_source))

    # 估算每线程寄存器数: b64 = 2 个 32-bit 寄存器
    total = f32_regs + b32_regs + 2 * b64_regs + pred_regs
    return {
        "f32": f32_regs, "b32": b32_regs, "b64": b64_regs,
        "pred": pred_regs, "estimated_total_32bit": total,
    }


def has_tensor_core(ptx_source: str) -> bool:
    """检查 PTX 是否使用了 Tensor Core (mma.sync 指令)。"""
    return "mma.sync" in ptx_source


# ══════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════


def main():
    print("=" * 70)
    print("  07 — PTX 汇编精读")
    print("=" * 70)

    print("""
  ╔══════════════════════════════════════════════════════════════════╗
  ║  PTX (Parallel Thread Execution) = NVIDIA GPU 的汇编语言        ║
  ║  这是 LLVM NVPTX backend 的产物，再经过 ptxas 变成 SASS 机器码    ║
  ╚══════════════════════════════════════════════════════════════════╝

  PTX 是你在不反汇编 SASS 的情况下能看到的最底层代码。
  从 PTX 中你可以直接判断:
    • 用了多少寄存器? (.reg 声明 → 寄存器压力)
    • 用了多少 shared memory? (.shared 声明)
    • 有没有用 Tensor Core? (mma.sync 指令)
    • 内存访问模式? (ld.global vs ld.shared)
    • 有哪些同步点? (bar.sync, membar)

  ⚠ 注意: PTX 还不是最终的机器码。ptxas (NVIDIA 汇编器) 会:
    • 做真正的寄存器分配 (可能会 spill)
    • 指令调度重排
    • 生成 SASS (Streaming Assembler，真正的机器码)
  所以 PTX 中的寄存器数是"需求"，实际可能被 spill。""")

    # ── 第一部分: Vector ops kernel ─────────────────────────
    print("\n" + "─" * 70)
    print("  第一部分: Vector ops kernel → PTX (算术 + 内存访问)")
    print("─" * 70)

    N = 1024
    x = torch.randn(N, device="cuda")
    y = torch.randn(N, device="cuda")
    out = torch.empty(N, device="cuda")
    kernel_vector_ops[(triton.cdiv(N, 256),)](x, y, out, N, BLOCK=256)
    torch.cuda.synchronize()

    ptx1 = find_latest_ptx()
    if ptx1:
        content = ptx1.read_text()
        regs = count_registers(content)
        print(f"\n  📄 PTX: {ptx1.name}")
        print(f"  📊 寄存器统计 (每线程):")
        print(f"     .f32: {regs['f32']} | .b32: {regs['b32']} | "
              f".b64: {regs['b64']} | .pred: {regs['pred']}")
        print(f"     估算 32-bit 寄存器总数: ~{regs['estimated_total_32bit']}")
        print(f"     (H100 每线程最多 255，但 >128 会显著降低 occupancy)")
        print(f"\n  📖 注释版 PTX (前 60 行):")
        print(f"  {'─' * 60}")
        annotated = annotate_ptx(content)
        for line in annotated.split("\n")[:60]:
            print(f"  {line}")
        if len(annotated.split("\n")) > 60:
            print(f"  ... (省略)")
    else:
        print("  ⚠  未找到 .ptx 文件")

    # ── 关键指令速查 ──────────────────────────────────────
    print("""
  ──────────────────────────────────────────────────────────────────
  PTX 关键指令速查
  ──────────────────────────────────────────────────────────────────

  📍 内存层次:
    ld.global.ca.f32  — 从 HBM 加载 (带 L1 cache, ~300-800 cycles)
    ld.global.cg.f32  — 从 HBM 加载 (绕过 L1, 用于 streaming)
    st.global.f32     — 写入 HBM (~300-800 cycles)
    ld.shared.f32     — 从 shared memory 加载 (~20-30 cycles)
    st.shared.f32     — 写入 shared memory (~20-30 cycles)

  📍 算术:
    add.f32           — 浮点加法
    mul.f32           — 浮点乘法
    fma.rn.f32        — 融合乘加 (a*b + c, 1 cycle, 四舍五入)
    max.f32           — 浮点 max
    div.approx.f32    — 浮点除法 (近似, 较快)

  📍 Tensor Core:
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    — Tensor Core MMA: M=16, N=8, K=16, A=f16 row, B=f16 col, C/D=f32

  📍 同步:
    bar.sync 0        — block 内所有线程同步 (__syncthreads)
    bar.warp.sync     — warp 内线程同步
    membar.cta        — CTA 级内存屏障 (保证 shared memory 可见性)""")

    # ── 第二部分: Matmul kernel ────────────────────────────
    print("\n" + "─" * 70)
    print("  第二部分: Matmul kernel → 寻找 Tensor Core 指令")
    print("─" * 70)

    M, N, K = 128, 128, 128
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float32)
    kernel_matmul_ptx[(1, 1)](A, B, C, M, N, K,
                               BLOCK_M=64, BLOCK_N=64, BLOCK_K=32)
    torch.cuda.synchronize()

    ptx2 = find_latest_ptx()
    if ptx2:
        content = ptx2.read_text()
        has_mma = has_tensor_core(content)
        print(f"\n  📊 Tensor Core (mma.sync) 检测: {'✅ 已启用' if has_mma else '❌ 未检测到'}")
        if has_mma:
            # 打印所有 mma 指令
            mma_lines = [l.strip() for l in content.split("\n") if "mma" in l]
            print(f"  MMA 指令 ({len(mma_lines)} 条):")
            for line in mma_lines[:5]:
                print(f"    {line}")
            if len(mma_lines) > 5:
                print(f"    ... 及 {len(mma_lines) - 5} 条")

        regs2 = count_registers(content)
        print(f"\n  📊 寄存器统计:")
        print(f"     估算 32-bit 寄存器总数: ~{regs2['estimated_total_32bit']}")
        print(f"     (MMA kernel 通常需要更多寄存器)")

    print("""
  🔑 PTX 阅读的实用技巧:

    1. 看寄存器数量: grep ".reg" *.ptx | wc -l
       太多 → 可能 spill → 试试减小 num_warps 或 BLOCK_SIZE

    2. 看是否有 mma.sync: grep "mma.sync" *.ptx
       没有 → tl.dot 没被识别为矩阵乘 → 检查维度和 dtype

    3. 看 shared memory: grep ".shared" *.ptx
       过大 → 可能限制 occupancy → 试试减小 num_stages

    4. 看同步点: grep "bar.sync\|membar" *.ptx
       过多 → 可能有隐式的 layout conversion

    5. 看全局内存访问: grep "ld.global\|st.global" *.ptx
       coalesced 访问时相邻线程访问相邻地址，ld.global 可以被合并。""")

    print("\n📖 下一步: python phase4_compiler/08_pass_pipeline.py")
    print("   看完整的 pass pipeline，理解每个 pass 做什么。\n")


if __name__ == "__main__":
    main()
