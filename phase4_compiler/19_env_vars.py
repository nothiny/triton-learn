"""
19_env_vars.py — Triton 环境变量速查手册

学习目标:
  1. 掌握所有 Triton 调试/性能分析相关的环境变量
  2. 知道每个变量的用途、输出、何时使用
  3. 建立"遇到问题 → 设置对应环境变量 → 诊断"的工作流

运行: python phase4_compiler/19_env_vars.py

前提: 已完成 01-18。
"""

import os
import sys
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════
# 完整的环境变量目录
# ══════════════════════════════════════════════════════════════════════


ENV_VAR_REFERENCE = {
    # ── IR Dump ──
    "IR Dump & Inspection": [
        {
            "var": "TRITON_KERNEL_DUMP",
            "values": "0|1",
            "desc": "Dump 所有编译阶段到 ~/.triton/cache/: .ttir, .ttgir, .ll, .ptx",
            "when": "想看编译器生成的 IR 时（最常用的调试变量）",
            "cost": "轻微 I/O 开销，不影响 kernel 性能",
        },
        {
            "var": "MLIR_PRINT_IR_AFTER_ALL",
            "values": "0|1",
            "desc": "每个 MLIR pass 之后都打印完整 IR（输出量巨大！）",
            "when": "深入 debug 某个 pass，想看 pass 前后的 IR 差异",
            "cost": "非常大的 stdout 输出，显著增加运行时间",
        },
        {
            "var": "TRITON_DUMP_IR",
            "values": "0|1",
            "desc": "类似 TRITON_KERNEL_DUMP，但输出格式不同",
            "when": "备选方案",
            "cost": "轻微",
        },
    ],

    # ── 编译控制 ──
    "Compilation Control": [
        {
            "var": "TRITON_KERNEL_OVERRIDE",
            "values": "0|1",
            "desc": "强制重新编译（跳过 JIT cache）",
            "when": "修改 kernel 源码后想看新的 IR/PTX",
            "cost": "每次运行都重新编译 → 第一次慢",
        },
        {
            "var": "TRITON_ALWAYS_COMPILE",
            "values": "0|1",
            "desc": "总是重新编译所有 kernel",
            "when": "开发中，确保用的是最新代码",
            "cost": "编译开销大",
        },
        {
            "var": "TRITON_CACHE_DIR",
            "values": "path",
            "desc": "覆盖 cache 目录位置（默认 ~/.triton/cache/）",
            "when": "需要隔离不同 Triton 版本的 cache",
            "cost": "无",
        },
        {
            "var": "TRITON_CACHE_LIMIT",
            "values": "bytes (e.g. 1073741824)",
            "desc": "限制 cache 目录大小，超出后清除旧条目",
            "when": "磁盘空间紧张",
            "cost": "可能清除有用的 cache → 需要重新编译",
        },
    ],

    # ── 性能分析 ──
    "Performance & Autotuning": [
        {
            "var": "TRITON_PRINT_AUTOTUNING",
            "values": "0|1",
            "desc": "打印 autotune 过程：测试了哪些 config，每个耗时多少",
            "when": "检查 autotuner 是否选了合理的 config",
            "cost": "轻微 stdout 开销",
        },
        {
            "var": "TRITON_ENABLE_AUTOTUNER",
            "values": "0|1",
            "desc": "开启/关闭 autotuner（0=用默认 config，不搜索）",
            "when": "快速测试，不想等 autotune",
            "cost": "关闭后性能可能差",
        },
        {
            "var": "TRITON_AUTOTUNE_EARLY_EXIT",
            "values": "0|1",
            "desc": "autotune 找到一个足够好的 config 后就退出",
            "when": "加速 autotune（可能错过更好的 config）",
            "cost": "可能找到次优 config",
        },
    ],

    # ── 调试 ──
    "Debugging & Interpretation": [
        {
            "var": "TRITON_INTERPRET",
            "values": "0|1",
            "desc": "CPU 解释执行 Triton kernel（不编译，不用 GPU）",
            "when": "调试 kernel 逻辑（支持 Python breakpoint!）",
            "cost": "速度慢 100-1000x，但支持 pdb.set_trace()",
        },
        {
            "var": "TRITON_KERNEL_DEBUG",
            "values": "0|1",
            "desc": "在 PTX 中保留更多调试信息 / 减少优化",
            "when": "用 cuda-gdb 调试 kernel",
            "cost": "性能下降",
        },
        {
            "var": "TRITON_MAX_TENSOR_NUMEL",
            "values": "整数 (默认 131072)",
            "desc": "解释器模式下允许的最大 tensor 元素数",
            "when": "TRITON_INTERPRET=1 模式下设置了过大的 tensor 导致 OOM",
            "cost": "只影响解释器模式",
        },
    ],

    # ── Driver / Runtime ──
    "Driver & Runtime": [
        {
            "var": "TRITON_PTXAS_PATH",
            "values": "path/to/ptxas",
            "desc": "指定 PTX 汇编器的路径 (ptxas)",
            "when": "系统有多个 CUDA 版本，需要指定 ptxas",
            "cost": "无",
        },
        {
            "var": "CUDA_VISIBLE_DEVICES",
            "values": "0,1,2,...",
            "desc": "指定 Triton 可见的 GPU（标准 CUDA 环境变量）",
            "when": "多 GPU 机器上指定用哪块 GPU",
            "cost": "无",
        },
        {
            "var": "TRITON_USE_LEGACY_LAUNCHER",
            "values": "0|1",
            "desc": "使用旧的 CUDA driver API 启动方式",
            "when": "遇到 driver 兼容问题时",
            "cost": "可能缺少新特性",
        },
    ],

    # ── 实验性 / 高级 ──
    "Experimental / Advanced": [
        {
            "var": "TRITON_CPU_BACKEND",
            "values": "0|1",
            "desc": "启用 Triton 的 CPU 后端（实验性）",
            "when": "在没有 GPU 的机器上测试",
            "cost": "CPU 性能远低于 GPU",
        },
        {
            "var": "TRITON_DISABLE_CACHE",
            "values": "0|1",
            "desc": "完全禁用 JIT cache",
            "when": "调试 cache 相关问题",
            "cost": "每次都重新编译",
        },
        {
            "var": "TRITON_MAX_SHARED_MEMORY",
            "values": "bytes (e.g. 49152)",
            "desc": "限制 shared memory 使用量",
            "when": "想强制降低 shared memory → 增加 occupancy",
            "cost": "可能降低性能",
        },
        {
            "var": "TRITON_ALLOW_IF",
            "values": "0|1",
            "desc": "允许在 kernel 中使用 if/else (旧版 Triton 默认关闭)",
            "when": "老版本 Triton 编译 if/else 报错时",
            "cost": "可能影响性能",
        },
    ],

    # ── LLVM / MLIR ──
    "LLVM / MLIR": [
        {
            "var": "LLVM_IR_ENABLE_DUMP",
            "values": "0|1",
            "desc": "Dump LLVM IR 到 stderr",
            "when": "只看 LLVM IR 时（比 TRITON_KERNEL_DUMP 更细粒度）",
            "cost": "输出量大",
        },
        {
            "var": "MLIR_ENABLE_DUMP",
            "values": "0|1",
            "desc": "Dump MLIR 到 stderr",
            "when": "只看 MLIR (TTIR/TTGIR) 时",
            "cost": "输出量大",
        },
    ],

    # ── Triton 版本相关 ──
    "Triton Version & Config": [
        {
            "var": "TRITON_OFFLINE_COMPILER",
            "values": "0|1",
            "desc": "使用离线编译器（不依赖 CUDA driver）",
            "when": "在 CI 环境中交叉编译",
            "cost": "需要预装所有依赖",
        },
    ],
}


# ══════════════════════════════════════════════════════════════════════
# 调试工作流
# ══════════════════════════════════════════════════════════════════════


DEBUGGING_WORKFLOWS = [
    {
        "scenario": "kernel 性能远低于预期",
        "steps": [
            "TRITON_KERNEL_DUMP=1 python my_kernel.py",
            "检查 .ttir → 确认 tl.dot 被识别",
            "检查 .ttgir → 看 convert_layout 数量",
            "检查 .ptx → 看有没有 mma.sync, 寄存器数量",
        ],
    },
    {
        "scenario": "autotune 选了一个很差的 config",
        "steps": [
            "TRITON_PRINT_AUTOTUNING=1 python my_kernel.py",
            "观察每个 config 的耗时",
            "检查被选中的 config 是否确实最快",
            "如果不对 → 增加 warmup/rep, 或用 prune 排除坏的 config",
        ],
    },
    {
        "scenario": "kernel 在 GPU 上崩溃 (cuda error)",
        "steps": [
            "TRITON_INTERPRET=1 python my_kernel.py",
            "在可疑代码处加 pdb.set_trace()",
            "解释器模式下可以逐步调试 tensor 的值",
            "确认逻辑正确后，回到 GPU 模式",
        ],
    },
    {
        "scenario": "每次运行都很慢 (编译时间)",
        "steps": [
            "检查 ~/.triton/cache/ 是否可写",
            "检查是否每次都在 autotune (用 TRITON_PRINT_AUTOTUNING=1 确认)",
            "如果 cache 有效但总被清除 → 检查 TRITON_CACHE_LIMIT",
            "确保 key 参数正确 (不要 key=['BLOCK_SIZE'] 之类的)",
        ],
    },
    {
        "scenario": "想对比不同 config 的 PTX",
        "steps": [
            "TRITON_KERNEL_DUMP=1 TRITON_KERNEL_OVERRIDE=1 python my_kernel.py",
            "修改 config → 再运行",
            "对比 ~/.triton/cache/<hash1>/*.ptx 和 <hash2>/*.ptx",
            "重点关注: .reg 数量, mma.sync 出现, shared memory 使用",
        ],
    },
    {
        "scenario": "怀疑寄存器 spilling",
        "steps": [
            "TRITON_KERNEL_DUMP=1 python my_kernel.py",
            "检查 .ptx: 搜索 'st.local' 或 'ld.local' → 如果出现, 一定有 spill",
            "检查 .reg 声明数量 → >200/线程 很可能 spill",
            "修复: 减小 BLOCK_SIZE, 用 fp16, 减少 num_warps",
        ],
    },
]


# ══════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════


def main():
    print("=" * 70)
    print("  19 — Triton 环境变量速查手册")
    print("=" * 70)

    # ── 当前环境 ────────────────────────────────────────
    print("─" * 70)
    print("  当前 Triton 相关环境变量:")
    print("─" * 70)
    triton_vars = {k: v for k, v in os.environ.items()
                   if k.startswith("TRITON_") or k.startswith("MLIR_")
                   or k.startswith("LLVM_") or k == "CUDA_VISIBLE_DEVICES"}
    if triton_vars:
        for k, v in sorted(triton_vars.items()):
            print(f"    {k}={v}")
    else:
        print("    (无)")
    print()

    # ── 完整参考 ────────────────────────────────────────
    for category, vars_list in ENV_VAR_REFERENCE.items():
        print(f"  ═══════════════════════════════════════════════════════")
        print(f"  {category}")
        print(f"  ═══════════════════════════════════════════════════════")
        for entry in vars_list:
            print(f"    {entry['var']}={entry['values']}")
            print(f"      {entry['desc']}")
            print(f"      何时用: {entry['when']}")
            print(f"      代价: {entry['cost']}")
            print()

    # ── 调试工作流 ──────────────────────────────────────
    print("  ═══════════════════════════════════════════════════════")
    print("  调试工作流速查")
    print("  ═══════════════════════════════════════════════════════")

    for wf in DEBUGGING_WORKFLOWS:
        print(f"\n  🔧 {wf['scenario']}")
        for i, step in enumerate(wf['steps'], 1):
            print(f"    {i}. {step}")

    # ── 最常用的 5 个 ───────────────────────────────────
    print("\n" + "─" * 70)
    print("  最常用的 5 个环境变量")
    print("─" * 70)
    print("""
  1. TRITON_KERNEL_DUMP=1
     → 任何时候想看 IR 时都用这个
     → 配合 TRITON_KERNEL_OVERRIDE=1 强制重编译

  2. TRITON_PRINT_AUTOTUNING=1
     → autotune 结果不符合预期时

  3. TRITON_INTERPRET=1
     → kernel 在 GPU 上崩溃，想在 CPU 上调试逻辑时

  4. MLIR_PRINT_IR_AFTER_ALL=1
     → 深入 debug 某个 pass 时 (输出量极大!)

  5. TRITON_CACHE_DIR=/tmp/triton_cache
     → 需要隔离不同 Triton 版本或项目的 cache 时

  常用组合:
    # 开发时 (强制重编译 + dump IR)
    TRITON_KERNEL_DUMP=1 TRITON_KERNEL_OVERRIDE=1 python my_kernel.py

    # 调试 autotune
    TRITON_PRINT_AUTOTUNING=1 TRITON_KERNEL_DUMP=1 python my_kernel.py

    # CPU 调试
    TRITON_INTERPRET=1 python -m pdb my_kernel.py
""")

    print("\n📖 下一步: python phase4_compiler/20_source_guide.py")
    print("   Triton 源码导航——从哪里开始读。\n")


if __name__ == "__main__":
    main()
