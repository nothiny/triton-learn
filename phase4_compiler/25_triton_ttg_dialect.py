"""
25_triton_ttg_dialect.py — Triton 的 `ttg` Dialect 完整参考

学习目标:
  1. 理解 ttg dialect 的所有 layout encoding 类型
  2. 掌握 ttg.convert_layout, ttg.async_copy 等 GPU 特有 op
  3. 理解 module-level attributes (num-warps, num-ctas, target)

运行: python phase4_compiler/25_triton_ttg_dialect.py

前提: 已完成 22-24。
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
# 生成多种 layout 的 TTGIR
# ══════════════════════════════════════════════════════════════════════


@triton.jit
def ttg_demo(x_ptr, out_ptr, M, N,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """
    包含 reduce + elementwise 的 kernel。
    会产生 blocked → slice → blocked 的 layout 转换。
    """
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    x = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :])
    # blocked layout

    row_sum = tl.sum(x, axis=1)
    # slice layout

    x_centered = x - row_sum[:, None]
    # blocked (x) ← slice (row_sum) → 需要 convert_layout!

    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :],
             x_centered)


@triton.jit
def ttg_matmul_demo(A, B, C, M, N, K,
                     BLOCK_M: tl.constexpr,
                     BLOCK_N: tl.constexpr,
                     BLOCK_K: tl.constexpr):
    """
    Matmul kernel，用于观察 dot_op 和 mma layout。
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
# TTGIR 分析工具
# ══════════════════════════════════════════════════════════════════════


def analyze_ttgir(ttgir_text: str) -> dict:
    """分析 TTGIR 的 structure。"""
    return {
        "module_attrs": dict(re.findall(
            r'("[\w.-]+")\s*=\s*([^\s,}]+)', ttgir_text
        )),
        "num_convert_layout": ttgir_text.count("convert_layout"),
        "num_async_copy": ttgir_text.count("async_copy"),
        "layouts": {
            "#blocked": ttgir_text.count("#ttg.blocked") + ttgir_text.count("#blocked<"),
            "#mma": ttgir_text.count("#ttg.mma") + ttgir_text.count("#mma<"),
            "#slice": ttgir_text.count("#ttg.slice") + ttgir_text.count("#slice<"),
            "#dot_op": ttgir_text.count("#ttg.dot_op") + ttgir_text.count("#dot_op<"),
        }
    }


def extract_layout_params(ttgir_text: str):
    """提取并解析 blocked layout 的参数。"""
    patterns = re.findall(
        r'#(?:ttg\.)?blocked<\{([^}]+)\}>', ttgir_text
    )
    results = []
    for p in patterns[:5]:  # 最多展示前 5 个
        params = {}
        for key in ["sizePerThread", "threadsPerWarp", "warpsPerCTA", "order"]:
            m = re.search(rf'{key}\s*=\s*\[([^\]]+)\]', p)
            if m:
                params[key] = [int(x.strip()) for x in m.group(1).split(",")]
        results.append(params)
    return results


# ══════════════════════════════════════════════════════════════════════
# ttg dialect 完整参考
# ══════════════════════════════════════════════════════════════════════

TTG_REFERENCE = {
    "Layout Encoding Types (★★★ 最重要的部分)": [
        {
            "type": "#ttg.blocked<{...}>",
            "params": "sizePerThread, threadsPerWarp, warpsPerCTA, order",
            "when": "所有 elementwise op 的输出",
            "notes": "标准分块布局。\n"
                     "线程在 tensor 的各维均匀分布。\n"
                     "90% 的 op 使用这个 layout。"
        },
        {
            "type": "#ttg.slice<{dim, parent}>",
            "params": "dim (规约维度), parent (父 layout)",
            "when": "tl.sum, tl.max 等规约操作的输出",
            "notes": "沿 dim 维度切片的布局。\n"
                     "规约轴的线程被'折叠'。\n"
                     "几乎总是需要 convert_layout 后才能用于 elementwise。"
        },
        {
            "type": "#ttg.mma<{versionMajor, versionMinor, instrShape}>",
            "params": "versionMajor=2 (Ampere/Hopper),\n"
                      "instrShape=[16,8,16] 或 [16,8,32]",
            "when": "tl.dot 的输出",
            "notes": "Tensor Core MMA 布局。\n"
                     "数据在 warp 内按 MMA fragment 分布。\n"
                     "instrShape 决定用哪个 MMA 指令。"
        },
        {
            "type": "#ttg.dot_op<{opIdx, parent}>",
            "params": "opIdx=0 (A 操作数) 或 1 (B 操作数),\n"
                      "parent (数据原来的 layout)",
            "when": "tl.dot 的输入 (如果可以原位转换)",
            "notes": "MMA 操作数布局。\n"
                     "opIdx=0 表示 A 矩阵, opIdx=1 表示 B 矩阵。\n"
                     "如果输入原本是 #blocked，编译器插入 convert_layout。"
        },
        {
            "type": "#ttg.scan<{...}>",
            "params": "dim, parent",
            "when": "tl.cumsum, tl.cumprod",
            "notes": "Scan/前缀和布局。Triton 2.1+ 支持。"
        },
    ],

    "Module-Level Attributes": [
        {
            "attr": "ttg.num-warps",
            "value": "4, 8, 16, ...",
            "notes": "每个 CTA 的 warp 数。决定 occupancy 和资源分配。"
        },
        {
            "attr": "ttg.num-ctas",
            "value": "1 (通常)",
            "notes": "CTA 数量。目前 Triton 几乎总是 1 (一个 kernel launch = 多个 CTA)。"
        },
        {
            "attr": "ttg.target",
            "value": '"cuda:80" (A100), "cuda:90" (H100), ...',
            "notes": "目标 GPU 架构。决定可用的 MMA 形状。"
        },
        {
            "attr": "ttg.threads-per-warp",
            "value": "32 (固定)",
            "notes": "每 warp 线程数。NVIDIA GPU 固定为 32。"
        },
    ],

    "GPU-Specific Operations": [
        {
            "op": "ttg.convert_layout",
            "desc": "改变 tensor 的 layout encoding",
            "cost": "可能: 免费 (warp shuffle) 或 shared memory round-trip",
            "notes": "★★★ 性能关键 op。\n"
                     "blocked → dot_op: 通常需要 shared memory。\n"
                     "blocked → slice: 规约后自然产生。\n"
                     "RemoveLayoutConversions pass 尝试消除冗余转换。"
        },
        {
            "op": "ttg.async_copy",
            "desc": "从 global memory 异步拷贝到 shared memory",
            "cost": "低 (不阻塞线程, 数据后台到达)",
            "notes": "Pipeline pass 插入。\n"
                     "使用 GPU 的 copy engine, 不占用 compute 资源。\n"
                     "对应 PTX: cp.async.ca.shared.global"
        },
        {
            "op": "ttg.async_wait",
            "desc": "等待之前的异步拷贝完成",
            "cost": "取决于数据是否已到达",
            "notes": "配合 async_copy 使用。\n"
                     "对应 PTX: cp.async.wait_group N"
        },
        {
            "op": "ttg.async_commit_group",
            "desc": "将之前的异步拷贝标记为一个 commit group",
            "cost": "几乎免费",
            "notes": "对应 PTX: cp.async.commit_group"
        },
        {
            "op": "ttg.local_alloc",
            "desc": "分配 shared memory buffer",
            "cost": "编译期 (占用 shared memory 配额)",
            "notes": "Pipeline pass 分配 shared memory buffer。\n"
                     "在 PTX 中表现为 .shared 声明。"
        },
    ],
}


# ══════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════


def main():
    print("=" * 70)
    print("  25 — Triton `ttg` Dialect 完整参考")
    print("=" * 70)

    # ── 生成 TTGIR (reduce + elementwise) ──────────────────
    print("─" * 70)
    print("  1. 生成包含 layout 转换的 TTGIR")
    print("─" * 70)

    M, N = 64, 128
    x2d = torch.randn(M, N, device="cuda")
    out2d = torch.empty(M, N, device="cuda")
    ttg_demo[(1,)](x2d, out2d, M, N, BLOCK_M=32, BLOCK_N=64)
    torch.cuda.synchronize()

    cache = Path.home() / ".triton" / "cache"
    ttgir_files = sorted(cache.rglob("*.ttgir"),
                          key=lambda p: p.stat().st_mtime, reverse=True)
    if not ttgir_files:
        print("  ⚠ 未找到 TTGIR 文件")
        return

    ttgir1 = ttgir_files[0].read_text()

    # ── 分析 TTGIR ──────────────────────────────────────
    analysis = analyze_ttgir(ttgir1)
    print(f"\n  TTGIR 结构分析:")
    print(f"    Module attributes: {analysis['module_attrs']}")
    print(f"    convert_layout 次数: {analysis['num_convert_layout']}")
    print(f"    async_copy 次数: {analysis['num_async_copy']}")
    print(f"    Layout 分布:")
    for layout, count in analysis['layouts'].items():
        if count > 0:
            print(f"      {layout}: {count}")

    # 提取 layout 参数
    layouts = extract_layout_params(ttgir1)
    if layouts:
        print(f"\n    解析到的 blocked layout 参数:")
        for i, lp in enumerate(layouts):
            print(f"      [{i+1}] sizePerThread={lp.get('sizePerThread')}, "
                  f"threadsPerWarp={lp.get('threadsPerWarp')}, "
                  f"warpsPerCTA={lp.get('warpsPerCTA')}, "
                  f"order={lp.get('order')}")

    # ── 生成 TTGIR (matmul) ─────────────────────────────
    print("\n" + "─" * 70)
    print("  2. 包含 MMA layout 的 TTGIR (matmul)")
    print("─" * 70)

    A = torch.randn(128, 128, device="cuda", dtype=torch.float16)
    B = torch.randn(128, 128, device="cuda", dtype=torch.float16)
    C = torch.empty(128, 128, device="cuda", dtype=torch.float32)
    ttg_matmul_demo[(1, 1)](A, B, C, 128, 128, 128,
                              BLOCK_M=64, BLOCK_N=64, BLOCK_K=32)
    torch.cuda.synchronize()

    ttgir_files2 = sorted(cache.rglob("*.ttgir"),
                           key=lambda p: p.stat().st_mtime, reverse=True)
    ttgir2 = ttgir_files2[0].read_text()
    analysis2 = analyze_ttgir(ttgir2)
    print(f"\n  MMA kernel TTGIR 分析:")
    print(f"    Layout 分布:")
    for layout, count in analysis2['layouts'].items():
        if count > 0:
            print(f"      {layout}: {count}")

    # 提取具体的 MMA 信息
    mma_attrs = re.findall(r'#ttg\.mma<\{([^}]+)\}>', ttgir2)
    dotop_attrs = re.findall(r'#ttg\.dot_op<\{([^}]+)\}>', ttgir2)
    if mma_attrs:
        print(f"    MMA layout: {mma_attrs[0]}")
    if dotop_attrs:
        print(f"    DotOperand layouts:")
        for da in dotop_attrs[:2]:
            print(f"      {da}")

    # ── ttg dialect 完整参考 ──────────────────────────────
    print("\n" + "═" * 70)
    print("  ttg Dialect 完整参考")
    print("═" * 70)

    for section_title, entries in TTG_REFERENCE.items():
        print(f"\n  ── {section_title} ──")
        for entry in entries:
            key = list(entry.keys())[0]
            name = entry[key]
            print(f"\n    ▸ {name}")
            if key == "type":
                print(f"      参数: {entry.get('params', '')}")
                print(f"      出现场景: {entry.get('when', '')}")
            elif key == "attr":
                print(f"      值: {entry.get('value', '')}")
            elif key == "op":
                print(f"      描述: {entry.get('desc', '')}")
                print(f"      开销: {entry.get('cost', '')}")
            if 'notes' in entry:
                for line in entry['notes'].strip().split('\n'):
                    print(f"      {line}")

    # ── TTGIR vs TTIR 关键差异总结 ───────────────────────
    print(f"""
  {'═' * 65}
  TTGIR vs TTIR: 关键差异
  {'═' * 65}

  ┌─────────────────────┬──────────────────────────────┐
  │ TTIR                │ TTGIR                        │
  ├─────────────────────┼──────────────────────────────┤
  │ tensor<256xf32>     │ tensor<256xf32,              │
  │                     │   #blocked<{...}>>           │
  │ 无线程信息            │ 明确的线程→数据映射            │
  │ 无 convert_layout   │ ttg.convert_layout 出现       │
  │ 无 async 操作        │ ttg.async_copy/async_wait    │
  │ 无 module 级别 GPU  │ ttg.num-warps, ttg.target   │
  │   属性               │                              │
  │ tt.dot 存在          │ #mma + #dot_op layout 出现   │
  │ tt.reduce 存在       │ #slice layout 出现           │
  └─────────────────────┴──────────────────────────────┘

  🔑 关键: TTGIR 并没有把 tt.load 替换成 ttg.load。
          它保留了 tt dialect 的 op，但在 TYPE 上加了 layout。
          只有少数新 op (convert_layout, async_copy) 是 ttg 独有的。
          这是 Triton MLIR 设计的精妙之处 —— type-driven lowering。""")

    print("\n📖 下一步: python phase4_compiler/26_mlir_pass_system.py")
    print("   深入 MLIR Pass 基础设施和 Triton 的 pass pipeline。\n")


if __name__ == "__main__":
    main()
