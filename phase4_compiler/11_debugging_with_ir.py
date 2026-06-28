"""
11_debugging_with_ir.py — 实战：用 IR 诊断性能问题

学习目标:
  1. 学会从 IR 中发现 4 种常见性能问题
  2. 建立 "性能异常 → 检查 IR → 定位问题" 的工作流
  3. 掌握每个问题对应的修复方案

运行: python phase4_compiler/11_debugging_with_ir.py

前提: 已运行 01-10，对整个编译管线有完整理解。
"""

import os
from pathlib import Path

os.environ["TRITON_KERNEL_DUMP"] = "1"
os.environ["TRITON_KERNEL_OVERRIDE"] = "1"

import torch
import triton
import triton.language as tl


# ══════════════════════════════════════════════════════════════════════
# 诊断案例 1: "我的 tl.dot 为什么没有用 Tensor Core?"
# ══════════════════════════════════════════════════════════════════════


@triton.jit
def case1_dot_fp32(A, B, C,
                    M, N, K,
                    BLOCK_M: tl.constexpr,
                    BLOCK_N: tl.constexpr,
                    BLOCK_K: tl.constexpr):
    """
    问题: A, B, C 都是 fp32 → tl.dot 可能不会触发 Tensor Core!
    Tensor Core 在 Ampere 上对 fp32 只支持 m16n8k8 (较小的 tile)。
    如果编译器判断不划算，可能退化为 elementwise 乘加。
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
        acc += tl.dot(a, b)           # fp32 × fp32 → 可能不触发 MMA!
    tl.store(C + rm[:, None] * N + rn[None, :], acc)


# ══════════════════════════════════════════════════════════════════════
# 诊断案例 2: "为什么 LayerNorm 很慢?" — layout conversion 过多
# ══════════════════════════════════════════════════════════════════════


@triton.jit
def case2_layernorm_naive(x_ptr, out_ptr, M, N,
                            BLOCK_M: tl.constexpr,
                            BLOCK_N: tl.constexpr):
    """
    一个naive的 LayerNorm 实现。
    多次在 blocked ↔ slice 之间切换 → 多次 convert_layout。

    问题: x → mean(x) [slice] → x - mean [转blocked] →
          var(x) [slice] → normalize [转blocked]
    每次 layout 切换都可能是 shared memory round-trip。
    """
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    x = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :])
    # blocked layout

    mean = tl.sum(x, axis=1) / BLOCK_N
    # slice layout (规约后的结果)

    x_centered = x - mean[:, None]
    # 需要 convert: blocked ← slice[:, None]
    # 这是第一次 convert_layout

    var = tl.sum(x_centered * x_centered, axis=1) / BLOCK_N
    # 又要 convert: slice ← blocked (规约输入)
    # 这是第二次 convert_layout

    rstd = 1.0 / tl.sqrt(var + 1e-5)
    # 仍在 slice layout

    x_norm = x_centered * rstd[:, None]
    # 又要 convert: blocked ← slice[:, None]
    # 这是第三次 convert_layout

    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], x_norm)


# ══════════════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════════════


def find_latest_ir(suffix):
    cache = Path.home() / ".triton" / "cache"
    if not cache.exists():
        return None
    files = sorted(cache.rglob(f"*.{suffix}"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def count_in_ir(ir_source, keyword):
    return ir_source.count(keyword)


# ══════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════


def main():
    print("=" * 70)
    print("  11 — 实战: 用 IR 诊断性能问题")
    print("=" * 70)

    # ═══════════════════════════════════════════════════════════
    # 诊断工作流概述
    # ═══════════════════════════════════════════════════════════
    print("""
  ╔══════════════════════════════════════════════════════════════════╗
  ║  性能诊断工作流:                                                ║
  ║                                                                 ║
  ║  1. 性能异常 (TFLOPS/带宽远低于预期)                             ║
  ║     ↓                                                           ║
  ║  2. Dump IR (TRITON_KERNEL_DUMP=1)                              ║
  ║     ↓                                                           ║
  ║  3. 检查 TTIR: tl.dot 有没有被识别?                              ║
  ║     ↓                                                           ║
  ║  4. 检查 TTGIR: 有多少 convert_layout? Layout 参数合理吗?        ║
  ║     ↓                                                           ║
  ║  5. 检查 PTX: 有 mma.sync 吗? 寄存器数量合理吗? 有 spill 吗?    ║
  ║     ↓                                                           ║
  ║  6. 定位问题 → 修复 → 重新 dump 验证                             ║
  ╚══════════════════════════════════════════════════════════════════╝""")

    # ═══════════════════════════════════════════════════════════
    # 案例 1: Missing MMA
    # ═══════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  诊断案例 1: tl.dot 没有触发 Tensor Core?")
    print("─" * 70)
    print("""
  症状: GEMM kernel 的 TFLOPS 远低于峰值
  怀疑: tl.dot 没有变成 MMA 指令

  Step 1: 检查 TTIR
    搜索 "tt.dot" → 如果在 TTIR 中，说明被识别为矩阵乘

  Step 2: 检查 PTX
    搜索 "mma.sync" → 这是 Tensor Core 的确定标志
    如果 TTIR 中有 tt.dot 但 PTX 中没有 mma.sync:
      → TritonGPUAccelerateMatmul pass 可能没有触发

  常见原因:
    ❌ fp32 输入 (Ampere MMA 对 fp32 支持有限)
    ❌ K 维度不是 16 的倍数
    ❌ BLOCK 大小与 MMA tile 不匹配
    ❌ 老版本 Triton 的 MMA 支持有问题

  修复:
    ✅ 用 fp16 或 bf16 输入
    ✅ 确保 K 是 16 的倍数 (Ampere: m16n8k16)
    ✅ 检查 BLOCK_K 是否 ≥ 16""")

    # 运行案例 1: fp32 dot
    print("\n  ▸ 运行案例 1: fp32 matmul")
    M, N, K = 128, 128, 128
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    C = torch.empty(M, N, device="cuda", dtype=torch.float32)
    case1_dot_fp32[(1, 1)](A, B, C, M, N, K, BLOCK_M=32, BLOCK_N=32, BLOCK_K=32)
    torch.cuda.synchronize()

    ptx1 = find_latest_ir("ptx")
    if ptx1:
        content = ptx1.read_text()
        has_mma = "mma.sync" in content
        print(f"     PTX 中有 mma.sync: {has_mma}")
        if has_mma:
            print(f"     → fp32 MMA 在 Ampere 上可用但 tile 较小 (m16n8k8)")
        else:
            print(f"     → 没有 MMA! tl.dot 退化为 elementwise 乘加。")

    # ═══════════════════════════════════════════════════════════
    # 案例 2: Too many convert_layout
    # ═══════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  诊断案例 2: convert_layout 过多")
    print("─" * 70)
    print("""
  症状: kernel 带宽利用率低，shared memory 使用异常高
  怀疑: 隐式的 layout conversion 太多

  Step 1: 检查 TTGIR
    搜索 "convert_layout" → 计数

  Step 2: 判断是否太多
    0-1 个: 正常
    2-3 个: 注意
    4+ 个: 可能有问题

  Step 3: 如果是 layer norm / softmax 类 kernel:
    这些 kernel 天然需要 blocked → slice → blocked 的切换
    但好的实现会尽量减少切换次数 (比如用 persistent layout)

  常见原因:
    ❌ 多次 blocked ↔ slice ↔ blocked 切换
    ❌ elementwise → reduce → elementwise → reduce 的模式
    ❌ blocked → mma → blocked → mma (可优化掉)

  修复:
    ✅ 合并所有 elementwise op 到一起 (减少 layout 切换)
    ✅ 使用专门的 kernel 设计模式 (layernorm, softmax 有专用实现)
    ✅ 检查 TTGIR, 看 convert 发生在哪里""")

    # 运行案例 2: naive LayerNorm
    print("\n  ▸ 运行案例 2: naive LayerNorm (预期较多 convert_layout)")
    M, N = 64, 256
    x2d = torch.randn(M, N, device="cuda")
    out2d = torch.empty(M, N, device="cuda")
    case2_layernorm_naive[(1,)](x2d, out2d, M, N, BLOCK_M=32, BLOCK_N=128)
    torch.cuda.synchronize()

    ttgir = find_latest_ir("ttgir")
    if ttgir:
        content = ttgir.read_text()
        n_convert = count_in_ir(content, "convert_layout")
        print(f"     TTGIR 中 convert_layout 数量: {n_convert}")
        if n_convert >= 3:
            print("     ⚠  有较多 convert_layout — 这可能影响性能")
        elif n_convert <= 1:
            print("     ✅ convert_layout 很少 — 编译器优化得不错")

    # ═══════════════════════════════════════════════════════════
    # 案例 3 & 4 总结
    # ═══════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  诊断案例 3 & 4: 寄存器 Spill 和 未 coalesced 访问")
    print("─" * 70)
    print("""
  案例 3: 寄存器 Spill

  症状: 性能突然下降，ncu 显示大量 local memory 访问
  诊断:
    PTX: 搜索 "st.local" 或 "ld.local" → 有就是 spill
    PTX: 看 .reg 数量 → >200/线程 很高了
  修复:
    → 减小 BLOCK_SIZE (减少每线程持有的元素)
    → 用 fp16 代替 fp32
    → 减少 num_warps (给每 warp 更多寄存器)

  案例 4: 未 coalesced 的内存访问

  症状: 带宽利用率远低于峰值
  诊断:
    PTX: 看 ld.global 的地址计算模式
    如果相邻线程的地址不是连续的 → 未 coalesced
  修复:
    → 确保 innermost 维度对应内存连续的维度
    → 调整 order 参数
    → 使用 tl.trans pose 或调整 load 的索引顺序""")

    # ═══════════════════════════════════════════════════════════
    # 诊断速查表
    # ═══════════════════════════════════════════════════════════
    print("─" * 70)
    print("  诊断速查表")
    print("─" * 70)
    print("""
  ┌────────────────────┬────────────────────┬─────────────────────────────────┐
  │ 症状                │ IR 中看到什么        │ 修复方案                          │
  ├────────────────────┼────────────────────┼─────────────────────────────────┤
  │ TFLOPS 远低于峰值   │ PTX 无 mma.sync      │ 换 fp16, 检查 K 维度, 更新 Triton │
  │ 带宽利用率低         │ TTGIR 多个 convert   │ 减少 layout 切换, 合并 elementwise │
  │ 性能突然下降         │ PTX 有 st.local      │ 减小 BLOCK, 用 fp16, 调 num_warps │
  │ Shared mem 超预期   │ TTGIR 大 .shared     │ 减小 num_stages, 减小 BLOCK       │
  │ Occupancy 低        │ PTX 多 .reg 声明     │ 同上 (寄存器或 shared mem 太大)    │
  │ Kernel 编译失败      │ 无 IR 输出           │ 检查错误信息, 检查 constexpr 参数  │
  │ Autotune 选差配置    │ 多份 PTX 差异大      │ 加更多 config, 调整 key 参数       │
  └────────────────────┴────────────────────┴─────────────────────────────────┘""")

    # ═══════════════════════════════════════════════════════════
    # 调试环境变量速查
    # ═══════════════════════════════════════════════════════════
    print("─" * 70)
    print("  调试环境变量速查")
    print("─" * 70)
    print("""
  TRITON_KERNEL_DUMP=1           → dump TTIR/TTGIR/LLVM/PTX 到 cache
  TRITON_KERNEL_OVERRIDE=1       → 强制重编译 (不用 cache)
  MLIR_PRINT_IR_AFTER_ALL=1      → 每个 pass 后打印 IR (输出量巨大)
  TRITON_ALWAYS_COMPILE=1        → 每次都重新编译
  TRITON_PRINT_AUTOTUNING=1      → 打印 autotune 测试了哪些 config
  TRITON_INTERPRET=1             → CPU 解释执行 (支持 Python 断点调试!)

  组合使用:
    TRITON_KERNEL_DUMP=1 TRITON_KERNEL_OVERRIDE=1 python my_kernel.py
    → 每次重新编译并 dump 所有 IR""")

    print("\n📖 下一步: python phase4_compiler/12_compile_api.py")
    print("   学习 triton.compiler API — 不用运行 kernel 就能拿到 IR。\n")


if __name__ == "__main__":
    main()
