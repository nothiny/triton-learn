"""
08_pass_pipeline.py — Pass Pipeline 全景

学习目标:
  1. 理解 Triton 编译器所有 pass 的名称和作用
  2. 知道哪些 pass 最关键，哪些是辅助优化
  3. 能用 MLIR_PRINT_IR_AFTER_ALL 观察每个 pass 的效果

运行: python phase4_compiler/08_pass_pipeline.py

前提: 已运行 01-07，对整个编译管线有基本认知。
"""

# 这个文件主要是概念讲解，不依赖 GPU 编译（但会演示如何开启 pass tracing）。

import os
import torch
import triton
import triton.language as tl


# ══════════════════════════════════════════════════════════════════════
# 一个简单的 kernel 用于演示
# ══════════════════════════════════════════════════════════════════════


@triton.jit
def demo_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
    """简单 kernel，用于观察 pass pipeline。"""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * 2.0 + y, mask=mask)


# ══════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════


def main():
    print("=" * 70)
    print("  08 — Pass Pipeline 全景")
    print("=" * 70)

    print("""
  ╔══════════════════════════════════════════════════════════════════╗
  ║  Pass = 编译器中对 IR 做一次变换的基本单元                       ║
  ║  每个 pass 做一件事: 优化、lowering、或者分析                     ║
  ╚══════════════════════════════════════════════════════════════════╝

  传统编译器 (LLVM/GCC): 几百个 pass，按固定顺序运行
  Triton 编译器: 约 20 个 pass，专注于 GPU 特殊优化

  你可以用 MLIR_PRINT_IR_AFTER_ALL=1 看到每个 pass 前后的 IR 变化。
  """)

    # ── 完整 Pass Pipeline ──────────────────────────────────
    print("─" * 70)
    print("  完整 Pass Pipeline (按执行顺序)")
    print("─" * 70)

    passes = [
        # ── Phase 1: TTIR 优化 (在纯 Triton IR 上) ──
        ("1. TTIR Preprocessing", [
            ("TritonInliner", "内联所有 @triton.jit 子函数调用"),
            ("TritonCombineOps", "融合相邻的 elementwise op (如 add+relu)"),
            ("TritonCanonicalize", "标准化 IR (常量折叠、死代码消除等)"),
        ]),

        # ── Phase 2: TTIR → TTGIR (最关键!) ──
        ("2. TTIR → TTGIR", [
            ("ConvertTritonToTritonGPU", "★★★ 最关键 pass! 分配 layout encoding"),
            ("TritonGPUCanonicalize", "GPU dialect 的标准化"),
        ]),

        # ── Phase 3: TTGIR 优化 ──
        ("3. TTGIR Optimizations", [
            ("TritonGPUCoalesce", "合并相邻线程的内存访问 → coalesced access"),
            ("TritonGPURemoveLayoutConversions", "消除冗余 layout 转换 (copy propagation)"),
            ("TritonGPUAccelerateMatmul", "★★ tl.dot → MMA intrinsic (Tensor Core)"),
            ("TritonGPUCombineTensorSelect", "优化 tensor select 操作"),
            ("TritonGPUReduceDataDuplication", "减少规约中的重复数据"),
        ]),

        # ── Phase 4: Pipeline & Prefetch ──
        ("4. Software Pipelining", [
            ("TritonGPUPipeline", "★★ Software pipeline: 展开循环+加载/计算重叠"),
            ("TritonGPUPrefetch", "★★ 插入 cp.async 预取指令"),
        ]),

        # ── Phase 5: TTGIR → LLVM ──
        ("5. Lowering to LLVM", [
            ("ConvertTritonGPUToLLVM", "TTGIR → LLVM IR (展开 layout, 生成 NVVM intrinsic)"),
        ]),

        # ── Phase 6: LLVM → PTX (LLVM 内置, 不是 Triton 控制) ──
        ("6. LLVM → PTX (NVPTX backend)", [
            ("LLVM Opt Pipeline", "LLVM 标准优化 (CSE, DCE, LICM, ...)"),
            ("NVPTX CodeGen", "寄存器分配 + 指令选择"),
            ("NVPTX Assembly Printer", "输出 PTX 文本"),
        ]),
    ]

    for phase_name, pass_list in passes:
        print(f"\n  ▸ {phase_name}")
        for pass_name, description in pass_list:
            stars = "★★★" if "★★★" in description else ("★★" if "★★" in description else "")
            desc_clean = description.replace("★★★ ", "").replace("★★ ", "")
            print(f"    {pass_name:40s} {desc_clean}")
            if stars:
                print(f"    {'':40s} {stars}")

    # ── 最关键的 5 个 Pass ──────────────────────────────────
    print("\n" + "─" * 70)
    print("  最关键的 5 个 Pass 详解")
    print("─" * 70)

    print("""
  ❶ ConvertTritonToTritonGPU (最重要!)
    ┌─────────────────────────────────────────────────────────────┐
    │ 输入: TTIR (纯数学描述, 无 GPU 信息)                         │
    │ 输出: TTGIR (每个 tensor 有 layout encoding)                 │
    │                                                             │
    │ 做出的决策:                                                  │
    │   • 每个 tensor 的 layout (blocked / slice / mma / dot_op)  │
    │   • sizePerThread / threadsPerWarp / warpsPerCTA 参数       │
    │   • 哪些地方需要 convert_layout                             │
    │                                                             │
    │ 类比: 寄存器分配 + 数据布局 + 指令调度的融合 pass             │
    │ 这是 Triton 编译器最独特的创新                                │
    └─────────────────────────────────────────────────────────────┘

  ❷ TritonGPUAccelerateMatmul
    ┌─────────────────────────────────────────────────────────────┐
    │ 识别 tt.dot 操作 → 替换为 MMA intrinsic                     │
    │                                                             │
    │ 自动选择:                                                    │
    │   A100 (SM80): m16n8k16 (f16) 或 m16n8k8 (f32)            │
    │   H100 (SM90): m16n8k32 (更大的 K tile)                    │
    │                                                             │
    │ 如果这个 pass 没有触发:                                      │
    │   → 检查 dtype (必须 fp16/bf16)                             │
    │   → 检查维度 (K 必须是 16 的倍数)                            │
    │   → 检查 Triton 版本                                        │
    └─────────────────────────────────────────────────────────────┘

  ❸ TritonGPUPipeline
    ┌─────────────────────────────────────────────────────────────┐
    │ Software pipelining: 让内存加载和计算重叠                    │
    │                                                             │
    │ num_stages=1: ─load─ ─compute─ ─load─ ─compute─            │
    │ num_stages=2: ─load─ ─load─ ─load─                         │
    │                  ─compute─ ─compute─ ─compute─              │
    │               (加载和计算在时间上重叠)                       │
    │                                                             │
    │ 实现: 展开循环 + 插入 cp.async + 管理 buffer 切换            │
    └─────────────────────────────────────────────────────────────┘

  ❹ TritonGPUPrefetch
    ┌─────────────────────────────────────────────────────────────┐
    │ 在循环中插入 prefetch 指令，提前把下一轮数据加载到 shared mem │
    │ 让计算单元在等数据时不会空转                                  │
    └─────────────────────────────────────────────────────────────┘

  ❺ TritonGPURemoveLayoutConversions
    ┌─────────────────────────────────────────────────────────────┐
    │ 消除冗余的 layout 转换 (类似 copy propagation)                │
    │                                                             │
    │ blocked → mma → blocked → mma                               │
    │          ↑______↑ 这一对可以消除!                            │
    │                                                             │
    │ 这个 pass 做得好坏直接影响性能。                              │
    └─────────────────────────────────────────────────────────────┘""")

    # ── 如何观察 Pass 效果 ──────────────────────────────────
    print("─" * 70)
    print("  实战: 如何观察每个 pass 的效果")
    print("─" * 70)
    print("""
  方法 1: MLIR_PRINT_IR_AFTER_ALL=1 (最详细)
    ```bash
    MLIR_PRINT_IR_AFTER_ALL=1 python my_kernel.py 2>&1 | less
    ```
    会打印每个 pass 之后的完整 IR。输出量巨大，适合深入 debug。

  方法 2: 只看特定 pass
    ```bash
    TRITON_KERNEL_DUMP=1 python my_kernel.py
    # 在 ~/.triton/cache/ 中找到 .ttir 和 .ttgir
    # .ttir = ConvertTritonToTritonGPU 之前的 IR
    # .ttgir = ConvertTritonToTritonGPU 之后的 IR
    # 对比这两个文件 → 看到最关键的变化
    ```

  方法 3: 对比不同 config 的产物
    ```bash
    # 运行 autotune，不同的 num_warps/nos_stages 会生成不同的 IR
    TRITON_KERNEL_DUMP=1 python my_autotuned_kernel.py
    # 在 cache 中找到不同 hash 的 PTX → 对比寄存器使用量
    ```

  💡 实用技巧:
    • 比较 .ttir 和 .ttgir → 看 layout encoding 是否合理
    • 比较 .ttgir 和 .ptx → 看是否用了 mma.sync
    • 比较不同 num_warps 的 .ptx → 看寄存器压力的差异""")

    # ── 运行 demo ────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  运行 demo kernel...")
    print("─" * 70)
    N = 256
    x = torch.randn(N, device="cuda")
    y = torch.randn(N, device="cuda")
    out = torch.empty(N, device="cuda")
    demo_kernel[(1,)](x, y, out, N, BLOCK=256)
    torch.cuda.synchronize()
    print("  ✅ Kernel 执行成功")

    print("""
  💡 试试这个:
    MLIR_PRINT_IR_AFTER_ALL=1 python phase4_compiler/08_pass_pipeline.py 2>&1 | head -200
    你会看到每个 pass 之后的 IR dump。""")

    print("\n📖 下一步: python phase4_compiler/09_pipeline_prefetch.py")
    print("   深入理解 Software Pipelining — Triton 最强大的优化 pass。\n")


if __name__ == "__main__":
    main()
