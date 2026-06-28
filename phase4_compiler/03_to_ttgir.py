"""
03_to_ttgir.py — 关键一步：TTIR → TTGIR

学习目标:
  1. 看到 layout encoding 如何"出现"在每个 tensor 上
  2. 理解 ConvertTritonToTritonGPU 这个 pass 做了什么决策
  3. 观察 layout encoding 的参数含义

运行: python phase4_compiler/03_to_ttgir.py

前提: 已运行 01 和 02，理解 TTIR 的基本结构。
"""

import os
import sys
from pathlib import Path

os.environ["TRITON_KERNEL_DUMP"] = "1"
os.environ["TRITON_KERNEL_OVERRIDE"] = "1"

import torch
import triton
import triton.language as tl


# ══════════════════════════════════════════════════════════════════════
# 定义几个 kernel，观察不同场景下的 layout
# ══════════════════════════════════════════════════════════════════════


@triton.jit
def kernel_1d_vector(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    """
    1D vector kernel — 最简单的 layout (BlockedEncoding 1D)
    观察: 只有一个维度的 blocked layout。
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * 2.0, mask=mask)


@triton.jit
def kernel_2d_rowwise(x_ptr, out_ptr, M, N,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """
    2D elementwise kernel — BlockedEncoding 2D
    观察: 两个维度的 blocked layout。
    编译器自动决定 threadsPerWarp × warpsPerCTA 的分配。
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    x = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :])
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], x + 1.0)


@triton.jit
def kernel_with_reduction(x_ptr, out_ptr, M, N,
                           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """
    Reduction kernel — 会产生 SliceEncoding
    观察: 规约操作沿某个轴减少维度，生成不同的 layout。
    """
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    x = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :])

    # 沿 axis=1 规约 → 输入是 (BLOCK_M, BLOCK_N) → 输出是 (BLOCK_M,)
    row_sum = tl.sum(x, axis=1)

    tl.store(out_ptr + offs_m, row_sum)


# ══════════════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════════════


def find_latest_ttgir():
    cache = Path.home() / ".triton" / "cache"
    if not cache.exists():
        return None
    files = sorted(cache.rglob("*.ttgir"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def explain_layout_in_ttgir(ttgir_source: str):
    """从 TTGIR 源码中提取并解释 layout encoding。"""
    import re

    # 找所有 #blocked<{...}> 出现
    blocked_pattern = r"#blocked<\{([^}]+)\}>"
    slice_pattern = r"#slice<\{([^}]+)\}>"

    blocked_attrs = re.findall(blocked_pattern, ttgir_source)
    slice_attrs = re.findall(slice_pattern, ttgir_source)

    explanations = []

    for attr_str in blocked_attrs:
        # 解析 sizePerThread, threadsPerWarp, warpsPerCTA, order
        results = {}
        for key in ["sizePerThread", "threadsPerWarp", "warpsPerCTA", "order"]:
            m = re.search(rf"{key}\s*=\s*\[([^\]]+)\]", attr_str)
            if m:
                results[key] = tuple(int(x.strip()) for x in m.group(1).split(","))
        explanations.append(("blocked", results))

    for attr_str in slice_attrs:
        results = {}
        for key in ["dim", "parent"]:
            m = re.search(rf"{key}\s*=\s*(\d+)", attr_str)
            if m:
                results[key] = int(m.group(1))
        explanations.append(("slice", results))

    return explanations


# ══════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════


def main():
    print("=" * 70)
    print("  03 — TTIR → TTGIR: Layout Encoding 的出现")
    print("=" * 70)

    print("""
  ╔══════════════════════════════════════════════════════════════════╗
  ║  这是整个 Triton 编译器最核心的一步                               ║
  ╚══════════════════════════════════════════════════════════════════╝

  TTIR 说的是"做什么运算"——纯数学描述。
  TTGIR 加上了"哪个线程处理哪些数据"——这是 GPU 最关心的问题。

  在 TTIR 中:  tensor<1024xf32>
                  ↑ 只知道是 1024 个 f32 元素的 tensor

  在 TTGIR 中: tensor<1024xf32, #blocked<{sizePerThread=[1],
                                            threadsPerWarp=[32],
                                            warpsPerCTA=[4],
                                            order=[0]}>>
                  ↑ 现在知道了:
                    • 每个线程处理 1 个元素 (sizePerThread=[1])
                    • 每个 warp 有 32 个线程 (threadsPerWarp=[32])
                    • 4 个 warp 合作 (warpsPerCTA=[4])
                    • 总共: 4×32×1 = 128 个元素/CUDA block

  这个 layout 信息是 ConvertTritonToTritonGPU 这个 pass 加上去的。
  它是 Triton 编译器最独特的创新——传统编译器没有这个层。""")

    # ── 示例 1: 1D vector kernel ──────────────────────────────
    print("\n" + "─" * 70)
    print("  示例 1: 1D Vector → TTGIR (最简单的 BlockedEncoding)")
    print("─" * 70)

    N = 1024
    x = torch.randn(N, device="cuda")
    out = torch.empty(N, device="cuda")
    kernel_1d_vector[(triton.cdiv(N, 128),)](x, out, N, BLOCK=128)
    torch.cuda.synchronize()

    ttgir1 = find_latest_ttgir()
    if ttgir1:
        content = ttgir1.read_text()
        print(f"\n  📄 TTGIR: {ttgir1.name}")
        print(f"  {'─' * 60}")
        for line in content.split("\n")[:60]:
            print(f"  {line}")
        if len(content.split("\n")) > 60:
            print(f"  ... (省略 {len(content.splitlines()) - 60} 行)")

        # 解析 layout
        info = explain_layout_in_ttgir(content)
        if info:
            print(f"\n  🔍 检测到的 layout encoding:")
            for ltype, attrs in info:
                if ltype == "blocked":
                    print(f"     BlockedEncodingAttr:")
                    spt = attrs.get("sizePerThread", ())
                    tpw = attrs.get("threadsPerWarp", ())
                    wpc = attrs.get("warpsPerCTA", ())
                    order = attrs.get("order", ())
                    print(f"       sizePerThread  = {spt}")
                    print(f"       threadsPerWarp = {tpw}")
                    print(f"       warpsPerCTA    = {wpc}")
                    print(f"       order          = {order}")
                    # 计算总元素数
                    total = 1
                    for i in range(len(spt)):
                        total *= spt[i] * tpw[i] * (wpc[i] if i < len(wpc) else 1)
                    print(f"       → 总元素/CTA ≈ {total}")

    print("""
  🔑 1D BlockedEncoding 解读:
    对于 1D tensor<128xf32>:
      sizePerThread=[1]:  每个线程持有 1 个 f32
      threadsPerWarp=[32]: warp 内 32 个线程沿 dim 0
      warpsPerCTA=[4]:    4 个 warp
      order=[0]:          只有一维，innermost

    分配: 元素 0→thread 0, 1→thread 1, ..., 31→thread 31 (warp 0)
          元素 32→thread 0 (warp 1), ... 以此类推""")

    # ── 示例 2: 2D kernel ────────────────────────────────────
    print("\n" + "─" * 70)
    print("  示例 2: 2D Elementwise → BlockedEncoding 2D")
    print("─" * 70)

    M, N = 128, 256
    x2d = torch.randn(M, N, device="cuda")
    out2d = torch.empty(M, N, device="cuda")
    kernel_2d_rowwise[(1, 1)](x2d, out2d, M, N, BLOCK_M=64, BLOCK_N=128)
    torch.cuda.synchronize()

    ttgir2 = find_latest_ttgir()
    if ttgir2:
        content = ttgir2.read_text()
        info = explain_layout_in_ttgir(content)
        print(f"\n  🔍 检测到的 2D layout encoding:")
        for ltype, attrs in info:
            if ltype == "blocked":
                spt = attrs.get("sizePerThread", ())
                tpw = attrs.get("threadsPerWarp", ())
                wpc = attrs.get("warpsPerCTA", ())
                order = attrs.get("order", ())
                print(f"     sizePerThread  = {spt}")
                print(f"     threadsPerWarp = {tpw}")
                print(f"     warpsPerCTA    = {wpc}")
                print(f"     order          = {order}")
                if len(spt) == 2 and len(tpw) == 2:
                    print(f"\n     解读 (示例, 可能不同):")
                    print(f"       dim 0 (M): {spt[0]}×{tpw[0]}×{wpc[0] if len(wpc)>0 else 1} = "
                          f"{spt[0]*tpw[0]*(wpc[0] if len(wpc)>0 else 1)} 元素/CTA")
                    print(f"       dim 1 (N): {spt[1]}×{tpw[1]}×{wpc[1] if len(wpc)>1 else 1} = "
                          f"{spt[1]*tpw[1]*(wpc[1] if len(wpc)>1 else 1)} 元素/CTA")
                    print(f"       order={order}: "
                          f"dim {order[0]} 是 innermost (内存连续)")

    print("""
  🔑 order 参数的含义:
    order=[0, 1] → dim 0 是 innermost (row-major 风格)
    order=[1, 0] → dim 1 是 innermost (column-major 风格)
    这影响着 coalescing: order 应该让 innermost 维对应内存中连续的维。""")

    # ── 示例 3: Reduction kernel ─────────────────────────────
    print("\n" + "─" * 70)
    print("  示例 3: Reduction → SliceEncoding")
    print("─" * 70)

    out_reduce = torch.empty(M, device="cuda")
    kernel_with_reduction[(1,)](x2d, out_reduce, M, N, BLOCK_M=32, BLOCK_N=64)
    torch.cuda.synchronize()

    ttgir3 = find_latest_ttgir()
    if ttgir3:
        content = ttgir3.read_text()
        # 找 slice encoding
        info = explain_layout_in_ttgir(content)
        types_found = set(ltype for ltype, _ in info)
        print(f"\n  🔍 检测到的 layout 类型: {types_found}")
        for ltype, attrs in info:
            print(f"     {ltype}: {attrs}")

    print("""
  🔑 SliceEncoding 的含义:
    当你做 tl.sum(x, axis=1) 时:
      输入: tensor<M×N, #blocked<...>>
      输出: tensor<M, #slice<{dim=1, parent=#blocked<...>}>>

    SliceEncoding 表示"从父 layout 沿 dim 方向切一片":
      • 规约维度 (dim=1) 上的线程不再持有独立元素
      • 它们通过 warp shuffle + shared memory 协作做 reduce
      • 最终每个"切片"只保留一行

  ⚠  SliceEncoding 通常需要 convert_layout → 这是额外的开销。""")

    # ── 总结 ──────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  总结: TTIR vs TTGIR")
    print("─" * 70)
    print("""
  ┌─────────────────┬──────────────────────────────────────────────┐
  │ TTIR            │ TTGIR                                        │
  ├─────────────────┼──────────────────────────────────────────────┤
  │ 纯数学描述        │ 加了线程→数据的映射                            │
  │ tensor<1024xf32> │ tensor<1024xf32, #blocked<{...}>>            │
  │ 无 GPU 信息      │ 内含 layout encoding                         │
  │ 无线程概念        │ 线程分配明确                                  │
  │ 各 op 是抽象的    │ 出现 ttg.convert_layout (数据重排)            │
  │ 可读性强          │ 加了硬件相关信息                              │
  └─────────────────┴──────────────────────────────────────────────┘

  编译器做的最关键决策:
    1. 每个 tensor 用什么 layout? (blocked / slice / mma / dot_operand)
    2. sizePerThread 多大? (影响寄存器用量)
    3. threadsPerWarp 怎么分? (影响 warp 内 coalescing)
    4. warpsPerCTA 怎么分? (影响 occupancy)
    5. 哪里需要 insert convert_layout? (可能有性能代价)

  这些决策在 ConvertTritonToTritonGPU 这个 pass 中一次性完成。
  它同时做了传统编译器中"寄存器分配 + 数据布局 + 指令调度"的工作。""")

    print("\n📖 下一步: python phase4_compiler/04_layout_system.py")
    print("   深入理解 5 种 layout 类型，学会读懂每一个参数。\n")


if __name__ == "__main__":
    main()
