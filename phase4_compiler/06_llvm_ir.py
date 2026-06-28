"""
06_llvm_ir.py — LLVM IR：寄存器、地址、分支

学习目标:
  1. 读懂 TTGIR→LLVM IR 转换后的关键变化
  2. 理解 LLVM IR 中的寄存器、地址计算、分支
  3. 看到 threadIdx / blockIdx 如何被显式读取

运行: python phase4_compiler/06_llvm_ir.py

前提: 已运行 01-05，熟悉 TTIR 和 TTGIR。
"""

import os
from pathlib import Path

os.environ["TRITON_KERNEL_DUMP"] = "1"
os.environ["TRITON_KERNEL_OVERRIDE"] = "1"

import torch
import triton
import triton.language as tl


# ══════════════════════════════════════════════════════════════════════
# 一个简单的 kernel 用于观察 LLVM IR
# ══════════════════════════════════════════════════════════════════════


@triton.jit
def simple_add(x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
    """
    最简单的 vector add，LLVM IR 会非常清爽。
    适合观察 TTGIR → LLVM 的关键转换。
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


# ══════════════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════════════


def find_latest_llvm_ir():
    cache = Path.home() / ".triton" / "cache"
    if not cache.exists():
        return None
    # Triton 3.x may not always write .ll files; search broadly
    for pattern in ["*.ll", "*.llir", "*llvm*"]:
        files = sorted(cache.rglob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if files:
            return files[0]
    # Fallback: search any recently modified file that might contain LLVM IR
    recent = sorted(cache.rglob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for f in recent[:30]:
        if f.is_file() and f.suffix not in ('.ttir', '.ttgir', '.ptx', '.cubin', '.json'):
            content = f.read_text(encoding='utf-8', errors='ignore')
            if 'define ' in content and 'target triple' in content:
                return f
    return None


def find_latest_ttgir():
    cache = Path.home() / ".triton" / "cache"
    if not cache.exists():
        return None
    files = sorted(cache.rglob("*.ttgir"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


# ══════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════


def main():
    print("=" * 70)
    print("  06 — LLVM IR: 从 TTGIR 到通用底层表示")
    print("=" * 70)

    print("""
  ╔══════════════════════════════════════════════════════════════════╗
  ║  TTGIR → LLVM IR: 发生了什么?                                   ║
  ╚══════════════════════════════════════════════════════════════════╝

  TTGIR 还有 Triton 特有的概念 (layout encoding, convert_layout, ...)。
  LLVM IR 是"通用底层表示" — 不特定于任何编程语言或硬件。

  ConvertTritonGPUToLLVM 这个 pass 做了:
    1. 展开 layout encoding → 显式的地址计算
       #blocked<{sizePerThread=[1], threadsPerWarp=[32], ...}>
       → threadId = threadIdx.x; offset = block_start + threadId
    2. 消除 ttg.convert_layout → warp shuffle / shared memory 操作
    3. 生成 NVVM intrinsic 调用 (读取 blockIdx, threadIdx 等)
    4. tt.dot → 展开为 LLVM intrinsic (MMA 相关)

  关键: Triton 编译器在 LLVM 之前不做寄存器分配。
        寄存器分配是 LLVM NVPTX backend 的工作。""")

    # ── 运行 kernel ────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  运行 simple_add kernel，生成 LLVM IR...")
    print("─" * 70)

    N = 256
    x = torch.randn(N, device="cuda")
    y = torch.randn(N, device="cuda")
    out = torch.empty(N, device="cuda")
    simple_add[(triton.cdiv(N, 64),)](x, y, out, N, BLOCK=64)
    torch.cuda.synchronize()

    # ── 阅读 LLVM IR ──────────────────────────────────────
    llvm_file = find_latest_llvm_ir()
    if llvm_file:
        content = llvm_file.read_text()
        print(f"\n  📄 LLVM IR: {llvm_file.name} ({len(content)} bytes)")
        print(f"  {'─' * 60}")
        for line in content.split("\n")[:80]:
            print(f"  {line}")
        if len(content.split("\n")) > 80:
            print(f"  ... (省略 {len(content.splitlines()) - 80} 行)")
    else:
        print("  ⚠  未找到 .ll 文件")
        return

    # ── 对比 TTGIR ────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  TTGIR → LLVM IR 的关键变化")
    print("─" * 70)

    ttgir_file = find_latest_ttgir()
    if ttgir_file:
        ttgir_content = ttgir_file.read_text()
        # 看看 TTGIR 中的 layout
        has_blocked = "#blocked" in ttgir_content
        has_convert = "convert_layout" in ttgir_content
        print(f"  TTGIR 中有 #blocked: {has_blocked}")
        print(f"  TTGIR 中有 convert_layout: {has_convert}")
        print(f"\n  TTIR/TTGIR 中: 还有 tensor<...xf32, #blocked<...>> 的概念")
        print(f"  LLVM IR 中:   只有 ptr, i32, float 等基础类型")
        print(f"                 layout encoding 已经完全展开为地址计算")

    # ── LLVM IR 关键特征讲解 ──────────────────────────────
    print("""
  ──────────────────────────────────────────────────────────────────
  LLVM IR 的关键特征 (用手动示例讲解)
  ──────────────────────────────────────────────────────────────────

  假设原始的 simple_add kernel 生成类似这样的 LLVM IR:

  ```llvm
  ; 1. 函数签名 — 注意参数是怎么传的
  define void @simple_add(
      ptr %x_ptr,           ; ← 指针！不再是 tensor
      ptr %y_ptr,
      ptr %out_ptr,
      i32 %N
  ) {
    ; 2. 读取 GPU 特殊寄存器 (通过 NVVM intrinsic)
    %ctaid_x = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()    ; blockIdx.x
    %tid_x   = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()      ; threadIdx.x

    ; 3. 地址计算 — 原来 layout encoding 做的事现在显式写成算术
    %block_start = mul i32 %ctaid_x, 64          ; pid * BLOCK_SIZE
    %offset = add i32 %block_start, %tid_x        ; pid*BLOCK + threadIdx

    ; 4. 边界检查
    %in_bounds = icmp slt i32 %offset, %N         ; offset < N ?

    ; 5. 指针偏移计算 (GEP = GetElementPtr)
    %addr_x = getelementptr float, ptr %x_ptr, i32 %offset
    %addr_y = getelementptr float, ptr %y_ptr, i32 %offset

    ; 6. 加载 (带条件: if in_bounds load, else undef)
    %val_x = load float, ptr %addr_x
    %val_y = load float, ptr %addr_y

    ; 7. 计算
    %result = fadd float %val_x, %val_y

    ; 8. 存储
    %addr_out = getelementptr float, ptr %out_ptr, i32 %offset
    store float %result, ptr %addr_out

    ret void
  }
  ```

  🔑 关键变化总结:

    TTGIR 中的:                     → LLVM IR 中的:
    ─────────────────────────────────────────────────────
    tensor<64xf32, #blocked<...>>   → ptr + 地址计算 (mul, add)
    tl.program_id(0)                 → nvvm.read.ptx.sreg.ctaid.x()
    tl.arange(0, BLOCK)             → nvvm.read.ptx.sreg.tid.x()
    mask (offsets < N)              → icmp slt
    tl.load(ptr + offs, mask=mask)  → getelementptr + load
    tl.store(ptr + offs, val, mask) → getelementptr + store
    tt.dot(a, b)                    → nvvm.mma.sync.* (Tensor Core intrinsic)

  🔑 LLVM IR 的特点:
    • 不再有 tensor 类型 — 一切都是 ptr, i32, float
    • 不再有 layout — 所有数据分布都展开了 (地址计算)
    • 不再有"block"概念 — 只有 ctaid/tid 寄存器读取
    • 仍然 target-independent: 这段 LLVM IR 还需要 NVPTX backend 才能变 PTX
    • SSA 形式: 每个 %var 只定义一次 (利于编译器分析和优化)""")

    # ── LLVM IR 中的 NVVM intrinsic ──────────────────────
    print("""
  ──────────────────────────────────────────────────────────────────
  NVVM Intrinsic 速查 (LLVM IR 中常见的 GPU 特殊寄存器访问)
  ──────────────────────────────────────────────────────────────────

  NVVM intrinsic 是 NVIDIA 为 LLVM 提供的 GPU 特殊功能接口:

  @llvm.nvvm.read.ptx.sreg.ctaid.x()    ← blockIdx.x
  @llvm.nvvm.read.ptx.sreg.ctaid.y()    ← blockIdx.y
  @llvm.nvvm.read.ptx.sreg.tid.x()      ← threadIdx.x
  @llvm.nvvm.read.ptx.sreg.tid.y()      ← threadIdx.y
  @llvm.nvvm.read.ptx.sreg.ntid.x()     ← blockDim.x
  @llvm.nvvm.read.ptx.sreg.nctaid.x()   ← gridDim.x
  @llvm.nvvm.barrier0()                 ← __syncthreads()
  @llvm.nvvm.mma.sync.*                 ← Tensor Core MMA 操作

  这些 intrinsic 在最终的 PTX 中会变成对应的特殊寄存器读取指令:
    mov.u32 %r1, %ctaid.x
    mov.u32 %r2, %tid.x""")

    print("\n📖 下一步: python phase4_compiler/07_ptx_assembly.py")
    print("   看 LLVM IR 如何变成 PTX GPU 汇编。\n")


if __name__ == "__main__":
    main()
