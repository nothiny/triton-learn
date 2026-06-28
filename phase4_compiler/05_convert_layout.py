"""
05_convert_layout.py — Layout 转换的代价

学习目标:
  1. 理解 convert_layout 为什么是"隐形的性能杀手"
  2. 看懂 TTGIR 中的 ttg.convert_layout op
  3. 知道如何减少不必要的 layout 转换

运行: python phase4_compiler/05_convert_layout.py

前提: 已运行 03 和 04，理解 layout encoding 系统。
"""

import os
from pathlib import Path

os.environ["TRITON_KERNEL_DUMP"] = "1"
os.environ["TRITON_KERNEL_OVERRIDE"] = "1"

import torch
import triton
import triton.language as tl


# ══════════════════════════════════════════════════════════════════════
# 场景 1: "好"的代码 — 最小化 layout conversion
# 如果一个 kernel 的所有 op 都在同一 layout 下，就不需要转换。
# ══════════════════════════════════════════════════════════════════════


@triton.jit
def good_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
    """
    "好"的 kernel: 全部是 elementwise op，所有 tensor 保持一致 layout。
    预期: TTGIR 中没有 (或极少) convert_layout。
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(x_ptr + offs, mask=mask)     # blocked layout
    y = tl.load(y_ptr + offs, mask=mask)     # 同样的 blocked layout
    z = x + y
    w = z * 2.0 + 1.0
    v = tl.maximum(w, 0.0)
    tl.store(out_ptr + offs, v, mask=mask)


# ══════════════════════════════════════════════════════════════════════
# 场景 2: "坏"的代码 — 频繁的 layout 转换
# 混合 elementwise 和 reduction 会导致 layout 不匹配。
# ══════════════════════════════════════════════════════════════════════


@triton.jit
def bad_kernel(x_ptr, out_ptr, M, N,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """
    这个 kernel 包含:
      1. elementwise (blocked layout)
      2. reduction (slice layout)
      3. 再 elementwise (需要把 slice 转回 blocked)
    → 会导致 convert_layout 出现。
    """
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    x = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :])
    # x 是 blocked layout

    x2 = x * x  # elementwise, 保持 blocked

    # 规约 → slice layout!
    norms = tl.sum(x2, axis=1)

    # 再用这个规约结果做 elementwise → 需要 convert_layout!
    # slice layout → blocked layout (或反过来)
    x_normalized = x / norms[:, None]

    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], x_normalized)


# ══════════════════════════════════════════════════════════════════════
# 场景 3: tl.dot 触发 layout conversion
# ══════════════════════════════════════════════════════════════════════


@triton.jit
def matmul_showing_conversion(A, B, C,
                               M, N, K,
                               BLOCK_M: tl.constexpr,
                               BLOCK_N: tl.constexpr,
                               BLOCK_K: tl.constexpr):
    """
    标准的 tiled matmul。观察 load 的 blocked layout →
    如何变成 dot 的 dot_op layout。
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
        a = tl.load(A_ptr + k)           # blocked layout
        b = tl.load(B_ptr + k * N)       # blocked layout
        acc += tl.dot(a, b)              # input: blocked → dot_op (convert!)
    tl.store(C + rm[:, None] * N + rn[None, :], acc)


# ══════════════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════════════


def find_latest_ttgir():
    cache = Path.home() / ".triton" / "cache"
    if not cache.exists():
        return None
    files = sorted(cache.rglob("*.ttgir"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def count_convert_layouts(ttgir_source):
    """统计 TTGIR 中 convert_layout 的次数。"""
    count = 0
    lines_with_convert = []
    for line in ttgir_source.split("\n"):
        if "convert_layout" in line:
            count += 1
            lines_with_convert.append(line.strip()[:120])
    return count, lines_with_convert


# ══════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════


def main():
    print("=" * 70)
    print("  05 — Layout 转换: 隐形的性能杀手")
    print("=" * 70)

    print("""
  ╔══════════════════════════════════════════════════════════════════╗
  ║  什么是 convert_layout?                                        ║
  ╚══════════════════════════════════════════════════════════════════╝

  当两个 op 的 layout 不匹配时，编译器自动插入 ttg.convert_layout:

    %a: tensor<128x64xf16, #blocked<{...}>>
    %b: tensor<128x64xf16, #mma<{...}>>
    %c = addf %a, %b    ← layout 不匹配！
    →
    %a_converted = ttg.convert_layout %a : #blocked → #mma
    %c = addf %a_converted, %b   ← 现在 layout 一致了

  convert_layout 的代价:
    • 可能只是 warp shuffle (warp 内线程交换数据，~5 个周期)
    • 可能是 shared memory round-trip (~20 个周期 + barrier 同步)
    • 最坏情况: 整个 CTA 需要同步 + 读/写 shared memory

  多少个 convert_layout 算"太多"?
    • 0-1 个: 正常
    • 2-3 个: 需要关注
    • 4+ 个: 很可能有性能问题
    • 10+ 个: 严重影响性能""")

    # ── 场景 1: Good kernel ────────────────────────────────
    print("\n" + "─" * 70)
    print("  场景 1: 「好」的 kernel — 纯 elementwise，无 convert")
    print("─" * 70)

    N = 1024
    x = torch.randn(N, device="cuda")
    y = torch.randn(N, device="cuda")
    out = torch.empty(N, device="cuda")
    good_kernel[(triton.cdiv(N, 256),)](x, y, out, N, BLOCK=256)
    torch.cuda.synchronize()

    good_ttgir = find_latest_ttgir()
    if good_ttgir:
        content = good_ttgir.read_text()
        n_convert, convert_lines = count_convert_layouts(content)
        print(f"  📊 convert_layout 出现次数: {n_convert}")
        if n_convert == 0:
            print("  ✅ 没有任何 layout 转换 — 所有 op 在同一 layout 下执行")
        else:
            for line in convert_lines:
                print(f"     {line}")

    print("""
  💡 "好"的代码特征:
    • 全部是 elementwise op (没有 reduction, 没有 dot)
    • 所有 tensor 从生到死都是一个 layout
    → 编译器不需要做任何数据重排""")

    # ── 场景 2: Bad kernel ─────────────────────────────────
    print("\n" + "─" * 70)
    print("  场景 2: 「坏」的代码 — 混合 elementwise + reduction")
    print("─" * 70)

    M, N = 64, 128
    x2d = torch.randn(M, N, device="cuda")
    out2d = torch.empty(M, N, device="cuda")
    bad_kernel[(1,)](x2d, out2d, M, N, BLOCK_M=32, BLOCK_N=64)
    torch.cuda.synchronize()

    bad_ttgir = find_latest_ttgir()
    if bad_ttgir:
        content = bad_ttgir.read_text()
        n_convert, convert_lines = count_convert_layouts(content)
        print(f"  📊 convert_layout 出现次数: {n_convert}")
        for line in convert_lines[:10]:
            print(f"     {line}")
        if n_convert > 3:
            print(f"  ⚠  出现 {n_convert} 次 convert_layout — 每次都可能需要 barrier + shared mem")

    print("""
  💡 "坏"的代码特征:
    • 夹杂 elementwise + reduction + elementwise
    • 每次切换 layout 都触发 convert_layout
    • 特例: norms = sum(x²) 然后 x/norms — 这是 LayerNorm 的简化版
    → 这就是为什么 Triton 的 LayerNorm 实现要特别小心 layout 管理""")

    # ── 场景 3: Matmul dot ─────────────────────────────────
    print("\n" + "─" * 70)
    print("  场景 3: tl.dot 触发的 layout conversion")
    print("─" * 70)

    K = 128
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float32)
    matmul_showing_conversion[(1, 1)](A, B, C, M, N, K,
                                       BLOCK_M=32, BLOCK_N=64, BLOCK_K=32)
    torch.cuda.synchronize()

    mm_ttgir = find_latest_ttgir()
    if mm_ttgir:
        content = mm_ttgir.read_text()
        n_convert, convert_lines = count_convert_layouts(content)
        print(f"  📊 convert_layout 出现次数: {n_convert}")
        for line in convert_lines[:10]:
            print(f"     {line}")
        if n_convert > 0:
            print(f"  ℹ  这些 convert_layout 是正常的:")
            print(f"     blocked layout (from load) → dot_op layout (for tl.dot)")
            print(f"     Triton 编译器优化 pass (RemoveLayoutConversions) 会尽量消除冗余")

    # ── 总结 ──────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  总结: 如何减少 layout conversion?")
    print("─" * 70)
    print("""
  1. 合并同 layout 的 op: 把 elementwise 放在一起做
     ✅ x = load(); y = x * 2 + 1; z = max(y, 0); store(z)
     ❌ x = load(); y = sum(x); z = x / y; store(z)  ← reduction 打断

  2. 注意 tl.dot 的输入 layout: 如果你的 load 后直接 dot，
     Triton 可能会"智能地"让 load 直接产生 dot_op layout。
     但这依赖于 Triton 版本和 kernel 写法。

  3. 集中做 reduction: 把需要 reduction 的计算集中在一起，
     减少 blocked ↔ slice 的切换次数。

  4. 避免"乒乓"效应: A(blocked) → B(slice) → C(blocked) → D(slice) ...
     每次切换都可能触发 convert_layout。

  5. Dump TTGIR 检查: 养成习惯，对性能敏感的 kernel 检查 TTGIR
     中的 convert_layout 数量。""")

    print("\n📖 下一步: python phase4_compiler/06_llvm_ir.py")
    print("   看 TTGIR 如何变成 LLVM IR——寄存器、地址、分支。\n")


if __name__ == "__main__":
    main()
