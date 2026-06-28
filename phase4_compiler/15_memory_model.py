"""
15_memory_model.py — Triton 内存模型深度解析

学习目标:
  1. 理解 Triton 的三种内存空间 (HBM / Shared Memory / Register)
  2. 跟踪一个 load 操作在各层 IR 中的形态变化
  3. 学会判断内存访问是否 coalesced
  4. 理解 shared memory bank conflict

运行: python phase4_compiler/15_memory_model.py

前提: 已完成 01-14。
"""

import os
from pathlib import Path

os.environ["TRITON_KERNEL_DUMP"] = "1"
os.environ["TRITON_KERNEL_OVERRIDE"] = "1"

import torch
import triton
import triton.language as tl


# ══════════════════════════════════════════════════════════════════════
# Kernel 1: 最简单的 global memory access
# ══════════════════════════════════════════════════════════════════════


@triton.jit
def simple_load(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    """
    最简单的全局内存加载 → 存储。
    观察各层 IR 如何表示内存访问。
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * 2.0, mask=mask)


# ══════════════════════════════════════════════════════════════════════
# Kernel 2: 不同的 cache modifier
# ══════════════════════════════════════════════════════════════════════


@triton.jit
def load_with_cache(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    """
    演示 tl.load 的 cache 修饰符。
    cache_modifier 影响生成的 PTX 中的 cache 操作符。
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    # [COMPILER] cache_modifier 影响 PTX 中的 cache 操作符:
    #   .ca → Cache All (通过 L1, 默认)
    #   .cg → Cache Global (绕过 L1, 直通 L2)
    #   .cs → Cache Streaming (streaming, 用于 write-only)
    #   .wt → Write-Through

    # 不指定 → 默认 .ca
    a = tl.load(x_ptr + offs, mask=mask, cache_modifier=".ca")
    # 绕过 L1 → .cg
    b = tl.load(x_ptr + offs, mask=mask, cache_modifier=".cg")

    tl.store(out_ptr + offs, a + b, mask=mask)


# ══════════════════════════════════════════════════════════════════════
# Kernel 3: Shared memory staging (coalesced vs uncoalesced)
# ══════════════════════════════════════════════════════════════════════


@triton.jit
def matmul_with_shared(A, B, C,
                        M, N, K,
                        BLOCK_M: tl.constexpr,
                        BLOCK_N: tl.constexpr,
                        BLOCK_K: tl.constexpr):
    """
    使用 shared memory 的 matmul。
    观察: global → shared → register 的数据流。
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # shared memory 分配 — 在 PTX 中表现为 .shared 声明
    a_sh = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float16)
    b_sh = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float16)

    A_ptr = A + rm[:, None] * K + rk[None, :]
    B_ptr = B + rk[:, None] * N + rn[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        # Load HBM → shared memory
        a_sh = tl.load(A_ptr + k)          # global load → 写到 register → 或直接到 shared
        b_sh = tl.load(B_ptr + k * N)

        # Load shared memory → register → MMA
        acc += tl.dot(a_sh, b_sh)          # dot 操作数在 register 中，通过 warp shuffle

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


def analyze_memory_in_ptx(ptx_source):
    """分析 PTX 中的内存指令分布。"""
    import re
    stats = {
        "ld.global": len(re.findall(r"ld\.global", ptx_source)),
        "st.global": len(re.findall(r"st\.global", ptx_source)),
        "ld.shared": len(re.findall(r"ld\.shared", ptx_source)),
        "st.shared": len(re.findall(r"st\.shared", ptx_source)),
        "cp.async": len(re.findall(r"cp\.async", ptx_source)),
        ".shared": len(re.findall(r"\.shared\s+", ptx_source)),
        "bar.sync": len(re.findall(r"bar\.sync", ptx_source)),
        "membar": len(re.findall(r"membar", ptx_source)),
    }
    return stats


# ══════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════


def main():
    print("=" * 70)
    print("  15 — Triton 内存模型深度解析")
    print("=" * 70)

    # ── 内存层次总览 ──────────────────────────────────────
    print("─" * 70)
    print("  1. GPU 内存层次与 Triton 的对应")
    print("─" * 70)
    print("""
  ╔══════════════════════════════════════════════════════════════════╗
  ║  GPU 内存层次                    Triton 中的表示                ║
  ╠══════════════════════════════════════════════════════════════════╣
  ║  HBM (Global Memory, ~80 GB)     tl.load / tl.store            ║
  ║   延迟: ~300-800 cycles          ptr + offset (指针算术)        ║
  ║   带宽: ~3 TB/s (H100)           cache_modifier: .ca/.cg/.cs   ║
  ║                                                                ║
  ║  L2 Cache (~50 MB, 共享)         (不可直接控制)                 ║
  ║   延迟: ~200 cycles              cache_modifier 间接影响        ║
  ║                                                                ║
  ║  Shared Memory (~228 KB/SM)      (隐式管理)                    ║
  ║   延迟: ~20-30 cycles            ttg.convert_layout 途经        ║
  ║   带宽: ~1.6 TB/s/SM             cp.async 用于 async copy       ║
  ║                                                                ║
  ║  L1 Cache (~256 KB/SM, 可配)     cache_modifier=.ca → 利用 L1  ║
  ║                                                                ║
  ║  Register File (65536/SM)        (LLVM 管理)                   ║
  ║   延迟: ~0 cycles (即时)         每个 tensor 元素 → 寄存器      ║
  ║   带宽: 无限 (不需要显式访问)     sizePerThread 控制            ║
  ╚══════════════════════════════════════════════════════════════════╝

  关键: Triton 程序员不直接管理 shared memory 和 register。
        编译器通过 layout encoding 自动管理这些。""")

    # ── 观察各层 IR 中的内存表示 ──────────────────────────
    print("\n" + "─" * 70)
    print("  2. 一个 load 在各层 IR 中的形态")
    print("─" * 70)

    N = 256
    x = torch.randn(N, device="cuda")
    out = torch.empty(N, device="cuda")
    simple_load[(1,)](x, out, N, BLOCK=256)
    torch.cuda.synchronize()

    print("""
  Python 代码:  x = tl.load(x_ptr + offs, mask=mask)

  TTIR 中的表示:
    %x = tt.load %ptr[%offsets] : tensor<256xf32>, %mask
    ↑ 只有"从内存加载 256 个 f32"的语义，不知道是 HBM 还是 shared memory

  TTGIR 中的表示:
    %x = tt.load %ptr[%offsets] : tensor<256xf32, #blocked<{...}>>
    ↑ 加了 layout encoding，知道了线程分配

  LLVM IR 中的表示:
    %addr = getelementptr float, ptr %x_ptr, i32 %offset
    %val = load float, ptr %addr
    ↑ 显式地址计算 + load 指令

  PTX 中的表示:
    ld.global.ca.f32 %f1, [%rd2];
    ↑ .global = HBM, .ca = 带 L1 cache
""")

    # ── Cache modifier 的影响 ────────────────────────────
    print("─" * 70)
    print("  3. cache_modifier 在 PTX 中的表现")
    print("─" * 70)

    load_with_cache[(1,)](x, out, N, BLOCK=128)
    torch.cuda.synchronize()

    ptx = find_latest_ir("ptx")
    if ptx:
        content = ptx.read_text()
        for mod in [".ca", ".cg", ".cs", ".wt"]:
            count = content.count(f"ld.global.{mod}")
            if count > 0:
                print(f"    ld.global.{mod}: {count} 次")

    print("""
  cache_modifier 详解:
    .ca (Cache All)       → 数据经过 L1 和 L2, 默认
    .cg (Cache Global)    → 绕过 L1, 只缓存到 L2
                            用于: 可能被其他 block 复用的数据
    .cs (Cache Streaming) → streaming, 不缓存
                            用于: write-only 数据 (只写一次)
    .wt (Write-Through)   → write-through 策略
                            用于: 需要立即对其他 SM 可见的数据

  实际效果 (以 H100 为例):
    .ca: ld.global.ca.f32 → 加载到 L1 + L2, 延迟 ~300 cycles (L1 hit: ~30)
    .cg: ld.global.cg.f32 → 绕过 L1, 只到 L2, 延迟 ~200 cycles (L2 hit)
    .cs: ld.global.cs.f32 → 不缓存, 每次访问 HBM ~800 cycles
""")

    # ── Coalesced Memory Access ───────────────────────────
    print("─" * 70)
    print("  4. Coalesced Memory Access (合并内存访问)")
    print("─" * 70)
    print("""
  关键概念: GPU 的 HBM 访问以"transaction"为单位 (32/64/128 bytes)。
  如果 warp 内 32 个线程访问的内存地址在同一个 transaction 范围内，
  GPU 可以合并这 32 次访问为 1 次 → 带宽利用率高。

  Coalesced (合并):
    thread 0 → addr 0x1000
    thread 1 → addr 0x1004
    thread 2 → addr 0x1008     ← 连续地址!
    ...
    thread 31 → addr 0x107C
    → 32 × 4 bytes = 128 bytes → 1 个 transaction 搞定
    → 带宽利用率: 100%

  Uncoalesced (未合并, strided):
    thread 0 → addr 0x1000
    thread 1 → addr 0x2000
    thread 2 → addr 0x3000     ← strided!
    ...
    → 32 个独立 transaction → 带宽利用率: ~3%

  Triton 如何保证 coalesced?
    • 默认的 BlockedEncoding 让相邻线程访问连续元素
    • sizePerThread[1]=4 → 线程处理 4 个连续元素 → coalesced!
    • sizePerThread[1]=1 且跨 stride → 可能未合并

  检验方法:
    在 PTX 中看 ld.global 后面的地址模式
    如果相邻线程的 ld.global 地址连续 → coalesced
""")

    # ── Shared Memory Bank Conflict ──────────────────────
    print("─" * 70)
    print("  5. Shared Memory Bank Conflict")
    print("─" * 70)

    # 运行 matmul kernel
    M, N, K = 128, 128, 128
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float32)
    matmul_with_shared[(1, 1)](A, B, C, M, N, K,
                                BLOCK_M=64, BLOCK_N=64, BLOCK_K=32)
    torch.cuda.synchronize()

    ptx2 = find_latest_ir("ptx")
    if ptx2:
        mem_stats = analyze_memory_in_ptx(ptx2.read_text())
        print(f"  内存指令统计:")
        for k, v in mem_stats.items():
            if v > 0:
                print(f"    {k}: {v}")

    print("""
  Shared Memory Bank Conflict:

  Shared memory 被分成 32 个 bank (对应 32 个 warp lane)。
  每个 bank 每个 cycle 只能服务一个地址。
  如果多个线程同时访问同一个 bank 的不同地址 → bank conflict → 串行化。

  例 (2-way bank conflict):
    thread 0 → shared[0]    ← bank 0
    thread 1 → shared[2]    ← bank 2
    ...
    thread 16 → shared[32]  ← bank 0 ← 冲突! 和 thread 0 同 bank
    thread 17 → shared[34]  ← bank 2 ← 冲突!
    → 2-way conflict → 2x 延迟

  例 (无 conflict, broadcast):
    thread 0 → shared[0]    ← bank 0
    thread 1 → shared[0]    ← bank 0 ← 同地址，broadcast → 无冲突!
    thread 2 → shared[0]    ← bank 0
    ...

  Triton 如何避免 bank conflict?
    • layout encoding 中的 padding (自动加 padding 到 shared memory)
    • BlockedEncoding 的 sizePerThread 选择
    • 但不总是完美 — 复杂的 access pattern 仍可能有 conflict
""")

    # ── 内存数据流图 ────────────────────────────────────
    print("─" * 70)
    print("  6. 一个 load→compute→store 的完整内存旅程")
    print("─" * 70)
    print("""
  ┌──────────────────────────────────────────────────────────┐
  │  Python:                                                  │
  │    x = tl.load(x_ptr + offs, mask=mask)                  │
  │    y = tl.load(y_ptr + offs, mask=mask)                  │
  │    z = x * y + x                                         │
  │    tl.store(out_ptr + offs, z, mask=mask)                │
  └──────────────┬───────────────────────────────────────────┘
                 │
    ┌────────────▼───────────────────────────────────────────┐
    │  TTIR:                                                  │
    │    %x = tt.load %x_ptr[%offs] : tensor<256xf32>        │
    │    %y = tt.load %y_ptr[%offs] : tensor<256xf32>        │
    │    %t = arith.mulf %x, %y : tensor<256xf32>            │
    │    %z = arith.addf %t, %x : tensor<256xf32>            │
    │    tt.store %out_ptr[%offs], %z : tensor<256xf32>      │
    │  (所有数据假定在某个共同空间，不区分 global/shared)      │
    └──────────────┬──────────────────────────────────────────┘
                 │ ConvertTritonToTritonGPU
    ┌────────────▼───────────────────────────────────────────┐
    │  TTGIR:                                                 │
    │    %x = tt.load ... : tensor<256xf32, #blocked<...>>   │
    │    %y = tt.load ... : tensor<256xf32, #blocked<...>>   │
    │    (layout encoding 决定了数据在 threads/warps 中的分布)│
    └──────────────┬──────────────────────────────────────────┘
                 │ ConvertTritonGPUToLLVM
    ┌────────────▼───────────────────────────────────────────┐
    │  LLVM IR:                                               │
    │    %addr_x = getelementptr ...                          │
    │    %x_reg = load float, ptr %addr_x                    │
    │    %addr_y = getelementptr ...                          │
    │    %y_reg = load float, ptr %addr_y                    │
    │    %t_reg = fmul float %x_reg, %y_reg                  │
    │    %z_reg = fadd float %t_reg, %x_reg                  │
    │    store float %z_reg, ptr %addr_out                   │
    │  (所有数据在"虚拟寄存器"中，LLVM 分配物理寄存器)         │
    └──────────────┬──────────────────────────────────────────┘
                 │ NVPTX CodeGen
    ┌────────────▼───────────────────────────────────────────┐
    │  PTX:                                                   │
    │    ld.global.ca.f32 %f1, [%rd2];  // HBM → register   │
    │    ld.global.ca.f32 %f3, [%rd4];  // HBM → register   │
    │    mul.f32 %f5, %f1, %f3;          // all in registers │
    │    add.f32 %f7, %f5, %f1;                              │
    │    st.global.f32 [%rd8], %f7;      // register → HBM  │
    │  (明确区分了 HBM (ld.global) 和 寄存器)                  │
    └──────────────────────────────────────────────────────────┘

  🔑 核心洞察:
    • HBM 访问只在 load 和 store 时发生
    • 计算完全在寄存器中进行 (x*y 不需要访问内存)
    • shared memory 在这个例子中没有用到
    • 如果有 reduction 或 layout conversion → 会插入 shared memory staging
""")

    print("\n📖 下一步: python phase4_compiler/16_mma_deep.py")
    print("   深入 MMA/Tensor Core 的内部机制。\n")


if __name__ == "__main__":
    main()
