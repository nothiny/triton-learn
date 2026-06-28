"""
27_ir_analysis_tools.py — 构建 IR 分析工具箱

学习目标:
  1. 构建可重用的 MLIR 文本分析工具
  2. 比较不同 config 的 IR 差异
  3. 将 IR 分析集成到日常开发工作流中

运行: python phase4_compiler/27_ir_analysis_tools.py

前提: 已完成 22-26。
"""

import os
import re
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any
from collections import Counter

os.environ["TRITON_KERNEL_DUMP"] = "1"
os.environ["TRITON_KERNEL_OVERRIDE"] = "1"

import torch
import triton
import triton.language as tl


# ══════════════════════════════════════════════════════════════════════
# Kernel 用于对比不同 config 的 IR
# ══════════════════════════════════════════════════════════════════════


@triton.jit
def analysis_target(x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    z = x * y + x + y
    tl.store(out_ptr + offs, z, mask=mask)


# ══════════════════════════════════════════════════════════════════════
# IR 分析工具箱
# ══════════════════════════════════════════════════════════════════════


@dataclass
class IROpStats:
    """单个 IR 文件的 op 统计。"""
    op_counts: Counter = field(default_factory=Counter)
    dialect_counts: Counter = field(default_factory=Counter)
    tensor_shapes: list[str] = field(default_factory=list)
    layout_types: set[str] = field(default_factory=set)
    register_estimate: int = 0
    shared_mem_bytes: int = 0
    num_convert_layout: int = 0
    num_async_copy: int = 0
    file_size: int = 0

    @classmethod
    def from_ir_text(cls, text: str) -> "IROpStats":
        stats = cls()
        stats.file_size = len(text)

        # 统计 op
        for m in re.finditer(r'(\w+)\.(\w+)', text):
            dialect = m.group(1)
            op_name = m.group(2)
            if not any(skip in dialect for skip in
                        ("param", "reg", "global", "visible", "entry")):
                stats.op_counts[f"{dialect}.{op_name}"] += 1
                stats.dialect_counts[dialect] += 1

        # 找 tensor shapes
        for m in re.finditer(r'tensor<([^>]+)>', text):
            stats.tensor_shapes.append(m.group(1))

        # 找 layout 类型
        if "#blocked" in text or "#ttg.blocked" in text:
            stats.layout_types.add("blocked")
        if "#mma" in text or "#ttg.mma" in text:
            stats.layout_types.add("mma")
        if "#slice" in text or "#ttg.slice" in text:
            stats.layout_types.add("slice")
        if "#dot_op" in text or "#ttg.dot_op" in text:
            stats.layout_types.add("dot_op")

        # convert_layout
        stats.num_convert_layout = text.count("convert_layout")
        stats.num_async_copy = text.count("async_copy")

        # 寄存器估算 (仅 PTX)
        for m in re.finditer(r'\.reg\s+\.(\w+)\s+', text):
            regtype = m.group(1)
            w = 2 if regtype in ("b64", "f64") else 1
            stats.register_estimate += w

        # shared memory (仅 PTX)
        for m in re.finditer(r'\.shared\s+.*?(\d+)\s*$', text, re.MULTILINE):
            try:
                stats.shared_mem_bytes += int(m.group(1))
            except ValueError:
                pass

        return stats


class IRComparator:
    """比较两个 IR 文件的差异。"""

    @staticmethod
    def compare(stats1: IROpStats, stats2: IROpStats,
                label1="Config A", label2="Config B") -> str:
        lines = []
        lines.append(f"\n  {'─' * 60}")
        lines.append(f"  IR 对比: {label1} vs {label2}")
        lines.append(f"  {'─' * 60}")

        # 文件大小
        lines.append(f"  文件大小: {stats1.file_size} → {stats2.file_size} "
                     f"({stats2.file_size - stats1.file_size:+d})")

        # 新增/消失的 op
        ops1 = set(stats1.op_counts.keys())
        ops2 = set(stats2.op_counts.keys())
        new_ops = ops2 - ops1
        removed_ops = ops1 - ops2
        if new_ops:
            lines.append(f"  新增 op: {new_ops}")
        if removed_ops:
            lines.append(f"  消失 op: {removed_ops}")

        # op 数量变化
        all_ops = ops1 | ops2
        changed = []
        for op in sorted(all_ops):
            c1 = stats1.op_counts.get(op, 0)
            c2 = stats2.op_counts.get(op, 0)
            if c1 != c2:
                changed.append(f"{op}: {c1} → {c2} ({c2-c1:+d})")
        if changed:
            lines.append(f"  Op 数量变化:")
            for c in changed:
                lines.append(f"    {c}")

        # Layout 变化
        if stats1.layout_types != stats2.layout_types:
            lines.append(f"  Layout 变化: {stats1.layout_types} → {stats2.layout_types}")

        # convert_layout
        if stats1.num_convert_layout != stats2.num_convert_layout:
            lines.append(f"  convert_layout: {stats1.num_convert_layout} → "
                         f"{stats2.num_convert_layout} "
                         f"({stats2.num_convert_layout - stats1.num_convert_layout:+d})")

        # 寄存器
        if stats1.register_estimate > 0 and stats2.register_estimate > 0:
            lines.append(f"  寄存器估算: {stats1.register_estimate} → "
                         f"{stats2.register_estimate} "
                         f"({stats2.register_estimate - stats1.register_estimate:+d})")

        # shared memory
        if stats1.shared_mem_bytes > 0 or stats2.shared_mem_bytes > 0:
            lines.append(f"  Shared memory: {stats1.shared_mem_bytes} → "
                         f"{stats2.shared_mem_bytes} "
                         f"({stats2.shared_mem_bytes - stats1.shared_mem_bytes:+d})")

        return "\n".join(lines)


def find_ir_in_cache(suffix: str, sort_by_time=True):
    """在 cache 中查找特定类型的 IR 文件。"""
    cache = Path.home() / ".triton" / "cache"
    if not cache.exists():
        return []
    if sort_by_time:
        return sorted(cache.rglob(f"*.{suffix}"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
    return list(cache.rglob(f"*.{suffix}"))


# ══════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════


def main():
    print("=" * 70)
    print("  27 — 构建 IR 分析工具箱")
    print("=" * 70)

    # ── 1. 分析单一 IR 文件 ────────────────────────────────
    print("─" * 70)
    print("  1. 分析单个 kernel 的 IR")
    print("─" * 70)

    N = 1024
    x = torch.randn(N, device="cuda")
    y = torch.randn(N, device="cuda")
    out = torch.empty(N, device="cuda")
    analysis_target[(triton.cdiv(N, 256),)](x, y, out, N, BLOCK=256)
    torch.cuda.synchronize()

    cache = Path.home() / ".triton" / "cache"
    ttir_files = sorted(cache.rglob("*.ttir"),
                         key=lambda p: p.stat().st_mtime, reverse=True)
    ttgir_files = sorted(cache.rglob("*.ttgir"),
                          key=lambda p: p.stat().st_mtime, reverse=True)
    ptx_files = sorted(cache.rglob("*.ptx"),
                        key=lambda p: p.stat().st_mtime, reverse=True)

    for label, files in [("TTIR", ttir_files), ("TTGIR", ttgir_files),
                           ("PTX", ptx_files)]:
        if files:
            f = files[0]
            stats = IROpStats.from_ir_text(f.read_text())
            print(f"\n  ▸ {label} ({f.name[:16]}...):")
            print(f"    文件大小: {stats.file_size} bytes")
            print(f"    Dialect 分布: {dict(stats.dialect_counts.most_common(5))}")
            if stats.tensor_shapes:
                # 去重
                unique_shapes = list(dict.fromkeys(stats.tensor_shapes))[:5]
                print(f"    Tensor shapes: {unique_shapes}")
            if stats.layout_types:
                print(f"    Layout 类型: {stats.layout_types}")
            if stats.num_convert_layout > 0:
                print(f"    convert_layout: {stats.num_convert_layout}")
            if stats.num_async_copy > 0:
                print(f"    async_copy: {stats.num_async_copy}")
            if stats.register_estimate > 0:
                print(f"    寄存器估算: ~{stats.register_estimate}")
            if stats.shared_mem_bytes > 0:
                print(f"    Shared memory: {stats.shared_mem_bytes} bytes")

    # ── 2. 对比不同 BLOCK_SIZE 的 IR ──────────────────────
    print("\n" + "─" * 70)
    print("  2. 对比不同 BLOCK 大小的 IR")
    print("─" * 70)

    # 重新运行 (但不同 BLOCK)
    out2 = torch.empty(N, device="cuda")
    analysis_target[(triton.cdiv(N, 512),)](x, y, out2, N, BLOCK=512)
    torch.cuda.synchronize()

    # 找两个最近的 TTIR
    recent_ttir = sorted(cache.rglob("*.ttir"),
                          key=lambda p: p.stat().st_mtime, reverse=True)[:2]
    if len(recent_ttir) >= 2:
        stats_blk256 = IROpStats.from_ir_text(recent_ttir[1].read_text())
        stats_blk512 = IROpStats.from_ir_text(recent_ttir[0].read_text())
        comparison = IRComparator.compare(
            stats_blk256, stats_blk512,
            "BLOCK=256", "BLOCK=512"
        )
        print(comparison)

    print("""
  🔑 BLOCK_SIZE 如何影响 IR:
    • TTIR: 几乎所有 op 都和 BLOCK_SIZE 有关 (make_range, tensor shape)
    • TTGIR: layout encoding 参数变化 (warpsPerCTA 可能不同)
    • PTX: 寄存器数量可能不同 (sizePerThread 变了)

  这种对比方法可以应用于:
    • 不同 num_warps 的对比
    • fp16 vs fp32 的对比
    • autotune 胜出 config vs 最差 config 的对比""")

    # ── 3. 统计整体 cache ──────────────────────────────────
    print("\n" + "─" * 70)
    print("  3. Cache 统计分析")
    print("─" * 70)

    all_ttir = find_ir_in_cache("ttir", sort_by_time=False)
    all_ttgir = find_ir_in_cache("ttgir", sort_by_time=False)
    all_ptx = find_ir_in_cache("ptx", sort_by_time=False)

    print(f"  Cache 中 IR 文件统计:")
    print(f"    .ttir:  {len(all_ttir)} 个")
    print(f"    .ttgir: {len(all_ttgir)} 个")
    print(f"    .ptx:   {len(all_ptx)} 个")

    # 统计所有 PTX 的寄存器使用分布
    if all_ptx:
        reg_counts = []
        for f in all_ptx[:20]:  # 只分析最近 20 个
            stats = IROpStats.from_ir_text(f.read_text())
            if stats.register_estimate > 0:
                reg_counts.append(stats.register_estimate)
        if reg_counts:
            print(f"\n  PTX 寄存器使用分布 (最近 20 个):")
            print(f"    最小: {min(reg_counts)}, 最大: {max(reg_counts)}")
            print(f"    平均: {sum(reg_counts)/len(reg_counts):.0f}")
            # 简单直方图
            buckets = {}
            for r in reg_counts:
                bucket = (r // 20) * 20
                buckets[bucket] = buckets.get(bucket, 0) + 1
            for bucket in sorted(buckets):
                bar = "█" * buckets[bucket]
                print(f"    {bucket:3d}-{bucket+19:3d}: {bar} ({buckets[bucket]})")

    # ── 4. 工具箱集成 ────────────────────────────────────
    print("\n" + "─" * 70)
    print("  4. IR 分析工具集成到工作流")
    print("─" * 70)
    print("""
  实用脚本模板 (保存为 analyze_ir.py):

  ```python
  from phase4_compiler.27_ir_analysis_tools import IROpStats, IRComparator
  import sys, os
  os.environ["TRITON_KERNEL_DUMP"] = "1"

  # 1. 运行你的 kernel
  # ... your kernel call here ...

  # 2. 分析
  from pathlib import Path
  cache = Path.home() / ".triton" / "cache"
  for suffix in ["ttir", "ttgir", "ptx"]:
      files = sorted(cache.rglob(f"*.{suffix}"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
      if files:
          stats = IROpStats.from_ir_text(files[0].read_text())
          print(f"\\n=== {suffix} ===")
          print(f"  Size: {stats.file_size} bytes")
          print(f"  Ops: {dict(stats.op_counts.most_common(5))}")
          print(f"  convert_layout: {stats.num_convert_layout}")
          if stats.register_estimate > 0:
              print(f"  Registers: ~{stats.register_estimate}")
          if stats.layout_types:
              print(f"  Layouts: {stats.layout_types}")
  ```

  🔑 核心工作流:
    1. 改代码 → 2. 运行 kernel → 3. 分析 IR → 4. 对比 IR → 5. 修复
    ←───────────────────────────────────────────────────────────┘
    (循环直到满意)

  关键检查点:
    • TTIR: tt.dot 存在吗? (Missing MMA 检测)
    • TTGIR: convert_layout 有几个? (Layout 效率)
    • PTX: mma.sync 存在吗? 寄存器数合理吗? (硬件利用率)
""")

    # ── MLIR 系列总结 ──
    print("─" * 70)
    print("  MLIR 系列总结 (22-27)")
    print("─" * 70)
    print("""
  22: MLIR 核心概念  — Operation, Type, Attribute, Dialect, Region
  23: MLIR 文本格式   — 语法详解, SSA 命名, use-def chain
  24: tt dialect    — 每个 op 的语义 + Python→TTIR 映射
  25: ttg dialect   — Layout encoding 类型 + GPU 特有 op
  26: Pass 系统      — Pipeline 结构 + Pattern Rewriting
  27: IR 分析工具    — 可重用的分析器, 对比器, cache 统计

  🎯 核心收获:
    • MLIR 不是一种语言，而是一个构建 IR 的框架
    • Triton 的 tt 和 ttg dialect 是 MLIR 框架的两个"插件"
    • Type-driven lowering: TTIR→TTGIR 的关键是给 TYPE 加 layout
    • Pattern Rewriting 是 MLIR pass 最核心的编程模式
    • 读懂 MLIR 文本 = 理解编译器的"内心独白"
""")

    print("\n🏁 MLIR 系列完成！")
    print("   Phase 4 总计: 27 个教程文件，覆盖从入门到源码分析。\n")


if __name__ == "__main__":
    main()
