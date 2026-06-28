"""
10_register_pressure.py — 寄存器分配与三资源约束

学习目标:
  1. 理解 GPU SM 的三种关键资源: 寄存器、Shared Memory、Warp 槽位
  2. 看懂 PTX 中的寄存器声明，判断是否有 spill 风险
  3. 理解 num_warps / BLOCK_SIZE / num_stages 如何影响资源消耗

运行: python phase4_compiler/10_register_pipeline.py

前提: 已运行 07-09，理解 PTX 和 pass pipeline。
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
# 不同"寄存器压力"的 kernel
# ══════════════════════════════════════════════════════════════════════


@triton.jit
def kernel_low_regs(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    """
    低保寄存器压力: 只有 1 个 load + 1 个 store。
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * 2.0, mask=mask)


@triton.jit
def kernel_high_regs(x_ptr, y_ptr, z_ptr, w_ptr, out_ptr, N, BLOCK: tl.constexpr):
    """
    高寄存器压力: 4 个 load + 多个中间变量 + 复杂计算。
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    # 加载 4 个输入 → 至少 4 个寄存器
    a = tl.load(x_ptr + offs, mask=mask)
    b = tl.load(y_ptr + offs, mask=mask)
    c = tl.load(z_ptr + offs, mask=mask)
    d = tl.load(w_ptr + offs, mask=mask)

    # 多个中间计算 → 更多寄存器
    t1 = a * b
    t2 = c + d
    t3 = t1 * t2
    t4 = t3 + a
    t5 = t4 * c
    t6 = t5 + t1
    result = t6 / (t6 + 1.0)

    tl.store(out_ptr + offs, result, mask=mask)


# ══════════════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════════════


def find_latest_ptx():
    cache = Path.home() / ".triton" / "cache"
    if not cache.exists():
        return None
    files = sorted(cache.rglob("*.ptx"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def analyze_registers(ptx_source: str) -> dict:
    """分析 PTX 中的寄存器使用。"""
    f32_count = len(re.findall(r"\.reg\s+\.f32\s+", ptx_source))
    b32_count = len(re.findall(r"\.reg\s+\.b32\s+", ptx_source))
    b64_count = len(re.findall(r"\.reg\s+\.b64\s+", ptx_source))
    pred_count = len(re.findall(r"\.reg\s+\.pred\s+", ptx_source))
    b16_count = len(re.findall(r"\.reg\s+\.b16\s+", ptx_source))
    # b64 占用 2 个 32-bit 寄存器
    total_32bit_equiv = f32_count + b32_count + b16_count + 2 * b64_count + pred_count

    # 查找 shared memory 分配
    shared_matches = re.findall(r"\.shared\s+\.align\s+(\d+)\s+\.b8\s+(\d+)", ptx_source)
    total_shared_bytes = sum(int(size) for _, size in shared_matches)

    return {
        "f32_regs": f32_count,
        "b32_regs": b32_count,
        "b64_regs": b64_count,
        "pred_regs": pred_count,
        "b16_regs": b16_count,
        "total_32bit_equiv": total_32bit_equiv,
        "shared_bytes": total_shared_bytes,
    }


# ══════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════


def main():
    print("=" * 70)
    print("  10 — 寄存器分配与三资源约束")
    print("=" * 70)

    # ── GPU SM 的资源模型 ──────────────────────────────────
    print("─" * 70)
    print("  GPU SM 的三资源约束模型")
    print("─" * 70)
    print("""
  每个 SM (Streaming Multiprocessor) 的资源是固定的:

  ┌──────────────────────┬─────────────────────────────┐
  │ 资源                  │ H100 (SM90) 每 SM 的总量     │
  ├──────────────────────┼─────────────────────────────┤
  │ 寄存器 (32-bit)       │ 65536                       │
  │ Shared Memory         │ 228 KB (可配置)              │
  │ 最大 Warps            │ 64                          │
  │ 最大 Thread Blocks    │ 32                          │
  └──────────────────────┴─────────────────────────────┘

  你的 kernel 每 CTA (block) 需要:
    寄存器: num_warps × 32 × registers_per_thread
    Shared Memory: .shared 声明中的总字节数
    Warps: num_warps

  如果任何一项超出 SM 总限制，occupancy 就会下降。

  💡 举例:
    如果每线程用 128 个寄存器，num_warps=8:
      每 CTA 寄存器: 8 × 32 × 128 = 32768
      H100 每 SM: 65536
      最多同时驻留: 65536 / 32768 = 2 个 CTA/SM

    如果 num_warps=4, 每线程用 255 个寄存器:
      每 CTA 寄存器: 4 × 32 × 255 = 32640
      最多同时驻留: 65536 / 32640 = 2 个 CTA/SM
      → 寄存器不是瓶颈! 瓶颈变成了 warp 数量太少。

  ⚠  什么是 register spilling?
    当 LLVM 分配的寄存器超过物理寄存器限制时:
      寄存器值溢出到 stack (存在 L1 cache 或 local memory)
      → 额外的 load/store → 性能下降 2-5x
      → 在 PTX 中表现为 st.local / ld.local 指令""")

    # ── 对比低/高寄存器压力 ───────────────────────────────
    print("─" * 70)
    print("  对比: 低寄存器压力 vs 高寄存器压力")
    print("─" * 70)

    N = 4096
    x = torch.randn(N, device="cuda")
    y = torch.randn(N, device="cuda")
    z = torch.randn(N, device="cuda")
    w = torch.randn(N, device="cuda")
    out = torch.empty(N, device="cuda")

    print("\n  ▸ kernel_low_regs (1 个 load)")
    kernel_low_regs[(triton.cdiv(N, 256),)](x, out, N, BLOCK=256)
    torch.cuda.synchronize()
    ptx_low = find_latest_ptx()
    if ptx_low:
        regs_low = analyze_registers(ptx_low.read_text())
        print(f"    估算 32-bit 寄存器/线程: ~{regs_low['total_32bit_equiv']}")
        print(f"    Shared memory/CTA: {regs_low['shared_bytes']} bytes")

    print("\n  ▸ kernel_high_regs (4 个 load + 复杂计算)")
    kernel_high_regs[(triton.cdiv(N, 256),)](x, y, z, w, out, N, BLOCK=256)
    torch.cuda.synchronize()
    ptx_high = find_latest_ptx()
    if ptx_high:
        regs_high = analyze_registers(ptx_high.read_text())
        print(f"    估算 32-bit 寄存器/线程: ~{regs_high['total_32bit_equiv']}")
        print(f"    Shared memory/CTA: {regs_high['shared_bytes']} bytes")
        if regs_low:
            diff = regs_high['total_32bit_equiv'] - regs_low['total_32bit_equiv']
            print(f"\n    🔺 高压力 kernel 多用 ~{diff} 个寄存器/线程")
            print(f"       对 num_warps=4: 4×32×{diff} = {4*32*diff} 额外寄存器/CTA")
            print(f"       对 num_warps=8: 8×32×{diff} = {8*32*diff} 额外寄存器/CTA")

    # ── Triton 如何间接控制寄存器 ──────────────────────────
    print("""
  ──────────────────────────────────────────────────────────────────
  Triton 如何间接控制寄存器使用?
  ──────────────────────────────────────────────────────────────────

  Triton 不直接做寄存器分配 (LLVM 做)，但通过以下参数间接影响:

  ┌──────────────────┬──────────────────────────────────────────┐
  │ 参数              │ 对寄存器的影响                             │
  ├──────────────────┼──────────────────────────────────────────┤
  │ num_warps ↑      │ 每 warp 可用的寄存器减少 (总池固定)         │
  │                  │ → 可能触发 LLVM 做更多 spill               │
  │                  │ → 但 occupancy 可能增加 (更多 warp 隐藏延迟) │
  ├──────────────────┼──────────────────────────────────────────┤
  │ sizePerThread ↑  │ 每线程持有更多元素 → 需要更多寄存器         │
  │ (BLOCK_SIZE ↑)   │ → 直接增加寄存器需求                       │
  ├──────────────────┼──────────────────────────────────────────┤
  │ num_stages ↑     │ 更多 pipeline buffer → 更多 shared memory │
  │                  │ → 间接影响 (挤占 occupancy)                │
  ├──────────────────┼──────────────────────────────────────────┤
  │ dtype             │ fp32 比 fp16 多用 1 倍寄存器宽度           │
  │                  │ 累加器建议用 tl.float32 (精度, 不是 register)│
  └──────────────────┴──────────────────────────────────────────┘

  🔑 寄存器压力的"甜蜜点":
    • < 64 寄存器/线程: 很轻松，可以考虑增加 num_warps
    • 64-128 寄存器/线程: 正常范围
    • 128-200 寄存器/线程: 较高，注意 occupancy
    • > 200 寄存器/线程: 很高，可能 spill，试试减小 BLOCK_SIZE
    • > 255 寄存器/线程: H100 上限! 一定会 spill""")

    # ── 实际测试: 不同 num_warps 对寄存器压力的影响 ──────
    print("─" * 70)
    print("  观察: num_warps 如何影响寄存器使用")
    print("─" * 70)
    print("""
  Triton 的 autotune 会自动测试不同 num_warps 配置。
  你可以通过对比不同 config 的 PTX 来看寄存器变化:

  ```python
  @triton.autotune(
      configs=[
          triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
          triton.Config({'BLOCK_SIZE': 128}, num_warps=8),
      ],
      key=['N']
  )
  @triton.jit
  def my_kernel(...):
      ...
  ```

  运行后，两个 config 会生成不同的 PTX → 可以对比 .reg 声明数量。

  通常在 cache 目录中:
    <hash_config1>/xxx.ptx  ← num_warps=4 的 PTX
    <hash_config2>/xxx.ptx  ← num_warps=8 的 PTX

  一般规律:
    num_warps ↑ → 每线程可用寄存器 ↓ → LLVM 可能 spill →
    → 但 occupancy ↑ → 综合效果取决于 kernel 是计算密集还是内存密集""")

    # ── 实用调试技巧 ──────────────────────────────────────
    print("─" * 70)
    print("  实用调试技巧")
    print("─" * 70)
    print("""
  1. 快速查看寄存器:
     grep ".reg" ~/.triton/cache/*.ptx | wc -l

  2. 检查是否有 spill (local memory):
     grep "st.local\|ld.local" ~/.triton/cache/*.ptx
     如果有输出 → spill 发生 → 性能有问题

  3. 估算 occupancy:
     每 SM 最大 warps = min(
        total_regs / (num_warps × 32 × regs_per_thread),
        total_shared / shared_per_cta,
        max_warps
      )

  4. 降低寄存器压力的方法:
     • 减小 BLOCK_SIZE → 减少 sizePerThread → 减少寄存器需求
     • 使用 fp16/bf16 代替 fp32 (寄存器宽度减半)
     • 减少同时活跃的中间变量 (但 Triton 会自己管理，你控制不了太多)
     • 增大 num_warps (但这可能触发更多 spill)

  5. 如果 autotune 选了一个看起来很差的 config:
     检查 TRITON_PRINT_AUTOTUNING=1 的输出
     看看它测试了哪些 config，每个的时间如何
     可能不是寄存器的问题，而是 shared memory 或 occupancy""")

    print("\n📖 下一步: python phase4_compiler/11_debugging_with_ir.py")
    print("   实战: 用 IR 诊断常见的性能问题。\n")


if __name__ == "__main__":
    main()
