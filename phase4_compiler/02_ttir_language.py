"""
02_ttir_language.py — TTIR (Triton IR) 详解

学习目标:
  1. 理解 TTIR 中每个 op 的含义
  2. 建立 "Python 代码 → TTIR op" 的映射关系
  3. 观察 constexpr 参数如何被"编译掉"

运行: python phase4_compiler/02_ttir_language.py

前提: 已运行 01_first_ir.py，知道 IR 文件在哪里。
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
# 示例 1: 最简单的 element-wise 操作
# ══════════════════════════════════════════════════════════════════════


@triton.jit
def example_elementwise(x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
    """
    包含多种 elementwise 操作的 kernel，用于展示 TTIR 的 op 多样性。
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)

    # 多样化的计算
    a = x + y                       # 加法
    b = x * y                       # 乘法
    c = a * 2.0 + 1.0              # FMA (融合乘加)
    d = tl.maximum(c, 0.0)          # ReLU
    e = d / (d + 1.0)               # 除法和加法

    tl.store(out_ptr + offs, e, mask=mask)


# ══════════════════════════════════════════════════════════════════════
# 示例 2: reduction — 会生成不同的 TTIR op
# ══════════════════════════════════════════════════════════════════════


@triton.jit
def example_reduce(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    """
    包含 reduction 和 broadcast 的 kernel。
    Reduction 是 TTIR 中的一个独立 op 类型。
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(x_ptr + offs, mask=mask)
    s = tl.sum(x)                         # tt.reduce (sum)
    tl.store(out_ptr + pid, s)            # scalar store


# ══════════════════════════════════════════════════════════════════════
# 示例 3: 矩阵乘法 — 核心 op: tt.dot
# ══════════════════════════════════════════════════════════════════════


@triton.jit
def example_dot(A, B, C,
                M, N, K,
                BLOCK_M: tl.constexpr,
                BLOCK_N: tl.constexpr,
                BLOCK_K: tl.constexpr):
    """
    矩阵乘法 tile。tt.dot 是 Triton 编译器中最被"特殊对待"的 op
    ——它会触发 Tensor Core MMA 指令的生成。
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    A_ptr = A + rm[:, None] * K + rk[None, :]
    B_ptr = B + rk[:, None] * N + rn[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(A_ptr + k)
        b = tl.load(B_ptr + k * N)
        acc += tl.dot(a, b)               # ← tt.dot — 这是最关键的 op!
    tl.store(C + rm[:, None] * N + rn[None, :], acc)


# ══════════════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════════════


def find_latest_ttir(kernel_name_hint=""):
    """找到最新的 TTIR 文件。"""
    cache = Path.home() / ".triton" / "cache"
    if not cache.exists():
        return None
    ttir_files = sorted(cache.rglob("*.ttir"), key=lambda p: p.stat().st_mtime, reverse=True)
    return ttir_files[0] if ttir_files else None


def annotate_ttir(ttir_source: str) -> str:
    """给 TTIR 源代码加上注释，解释每个关键 op。"""
    annotations = {
        "tt.load":     "  ← 从内存加载一个 tensor（会变成 GPU 的 load 指令）",
        "tt.store":    "  ← 存储一个 tensor 到内存（会变成 GPU 的 store 指令）",
        "tt.dot":      "  ← 矩阵乘法（会触发 Tensor Core MMA！）",
        "tt.reduce":   "  ← 规约操作（sum, max, min 等沿某个轴）",
        "tt.broadcast":"  ← 广播：把一个标量/小 tensor 扩展成大 tensor",
        "tt.arange":   "  ← 生成等差数列 [0, 1, 2, ..., N-1]",
        "tt.make_range":" ← 同 arange，生成一个范围",
        "tt.program_id":" ← 获取当前 block 的索引（blockIdx.x/y/z）",
        "arith.addf":  "  ← 浮点加法",
        "arith.mulf":  "  ← 浮点乘法",
        "arith.divf":  "  ← 浮点除法",
        "arith.maximumf":" ← 浮点 max",
        "arith.addi":  "  ← 整数加法（常用于地址计算）",
        "arith.muli":  "  ← 整数乘法",
        "arith.cmpi":  "  ← 整数比较（生成 mask）",
        "arith.select":"  ← 条件选择（类似 C 的 ?:）",
    }
    lines = ttir_source.split("\n")
    annotated = []
    for line in lines:
        annotated.append(line)
        for key, note in annotations.items():
            if key in line and key not in line[line.index(key) - 1:line.index(key)]:
                # 简单检查：op 名出现在这一行
                if key in line:
                    annotated.append(f"                                            {note}")
                    break
    return "\n".join(annotated)


# ══════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════


def main():
    print("=" * 70)
    print("  02 — TTIR (Triton IR) 详解")
    print("=" * 70)

    # ── 示例 1: Elementwise kernel ──────────────────────────────
    print("\n" + "─" * 70)
    print("  示例 1: Elementwise kernel → TTIR")
    print("─" * 70)
    print("""
  你的 Python 代码:
    x = tl.load(x_ptr + offs, mask=mask)     # 加载
    y = tl.load(y_ptr + offs, mask=mask)
    a = x + y                                 # 加法
    b = x * y                                 # 乘法
    c = a * 2.0 + 1.0                         # FMA
    d = tl.maximum(c, 0.0)                    # ReLU
    tl.store(out_ptr + offs, d, mask=mask)    # 存储

  编译器会把它翻译成 TTIR。TTIR 是 MLIR (Multi-Level IR) 的一种 dialect，
  叫做 "tt" dialect。MLIR 使用 SSA (Static Single Assignment) 形式
  ——每个值有且仅有一次定义，通过 %0, %1, %2... 引用。""")

    N = 1024
    x = torch.randn(N, device="cuda")
    y = torch.randn(N, device="cuda")
    out = torch.empty(N, device="cuda")
    example_elementwise[(triton.cdiv(N, 64),)](x, y, out, N, BLOCK=64)
    torch.cuda.synchronize()

    ttir = find_latest_ttir()
    if ttir:
        content = ttir.read_text()
        # 只打印度量 elementwise 相关的部分
        print(f"\n  📄 最新 TTIR: {ttir.name}")
        print(f"  {'─' * 60}")
        for line in content.split("\n")[:70]:
            print(f"  {line}")
        if len(content.split("\n")) > 70:
            print(f"  ... (省略 {len(content.splitlines()) - 70} 行)")

    print("""
  🔑 从 Python 到 TTIR 的映射:
    tl.load()     → tt.load       加载 tensor
    tl.store()    → tt.store      存储 tensor
    tl.arange()   → tt.make_range 生成 [0, 1, ..., N-1]
    tl.program_id → tt.get_program_id  获取 block 索引
    x + y         → arith.addf    浮点加法
    x * y         → arith.mulf    浮点乘法
    offs < N      → arith.cmpi    整数比较 → 生成 mask""")

    # ── 示例 2: Reduction kernel ──────────────────────────────
    print("\n" + "─" * 70)
    print("  示例 2: Reduction kernel → TTIR")
    print("─" * 70)
    print("""
  你的 Python 代码:
    x = tl.load(x_ptr + offs, mask=mask)
    s = tl.sum(x)           ← 规约操作
    tl.store(out_ptr + pid, s)

  注意: tl.sum() 不是普通的 arith op，它会生成一个 tt.reduce op。
  Reduce 和 elementwise 在编译器中走不同的路径。""")

    out_scalar = torch.empty(1, device="cuda")
    example_reduce[(1,)](x, out_scalar, N, BLOCK=1024)
    torch.cuda.synchronize()

    ttir2 = find_latest_ttir()
    if ttir2:
        content = ttir2.read_text()
        print(f"\n  📄 最新 TTIR: {ttir2.name}")
        print(f"  {'─' * 60}")
        print(f"  (只显示与 reduce 相关的行)")
        for line in content.split("\n"):
            if any(kw in line for kw in ["tt.reduce", "tt.load", "tt.store"]):
                print(f"  {line}")

    print("""
  🔑 Reduce 在 TTIR 中的样子:
    tt.reduce(%input) { axis = 0 : i32 } → reduction 在 axis=0 上做
    reduction 的具体操作在 lambda 中指定（如 addf）

  🔑 关键区别:
    • arith.addf:    两个标量/tensor 相加 → 输出和输入一样大
    • tt.reduce:     沿某个 axis 规约 → 输出比输入少一个维度""")

    # ── 示例 3: Matmul kernel ─────────────────────────────────
    print("\n" + "─" * 70)
    print("  示例 3: Matmul kernel → TTIR (最关键: tt.dot)")
    print("─" * 70)
    print("""
  你的 Python 代码:
    acc += tl.dot(a, b)    ← 这个 tl.dot 是 Triton 最重要的 op

  tl.dot 在 TTIR 中是 tt.dot — 它告诉编译器:
    "这里需要矩阵乘法，请用 Tensor Core 来做"

  如果编译器识别不到 tt.dot（比如因为维度不对），
  就会退化成 elementwise 乘加序列 → 性能下降 3-5x。""")

    M, N, K = 128, 128, 128
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float32)
    example_dot[(1, 1)](A, B, C, M, N, K, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32)
    torch.cuda.synchronize()

    ttir3 = find_latest_ttir()
    if ttir3:
        content = ttir3.read_text()
        print(f"\n  📄 最新 TTIR: {ttir3.name}")
        print(f"  {'─' * 60}")
        for line in content.split("\n"):
            if any(kw in line for kw in ["tt.dot", "tt.load", "tt.store", "scf.for"]):
                print(f"  {line}")
        print()

    # ── 总结 ──────────────────────────────────────────────────
    print("─" * 70)
    print("  TTIR op 速查表")
    print("─" * 70)
    print("""
  ┌──────────────────────┬──────────────────────────┬────────────────────────────────┐
  │ TTIR op              │ 对应的 Python/Triton 代码   │ 编译器如何处理                   │
  ├──────────────────────┼──────────────────────────┼────────────────────────────────┤
  │ tt.load              │ tl.load(ptr, mask=...)   │ → GPU load 指令               │
  │ tt.store             │ tl.store(ptr, val, mask) │ → GPU store 指令              │
  │ tt.dot               │ tl.dot(a, b)             │ → Tensor Core MMA (关键!)     │
  │ tt.reduce            │ tl.sum(x), tl.max(x)     │ → warp shuffle + shared mem   │
  │ tt.broadcast         │ (隐式，如 scalar+tensor)   │ → 数据复制到各线程             │
  │ tt.make_range        │ tl.arange(0, N)          │ → 生成 [0,1,...,N-1]          │
  │ tt.get_program_id    │ tl.program_id(axis)      │ → 读取 blockIdx 寄存器         │
  │ arith.addf/mulf/divf │ x + y, x * y, x / y      │ → GPU add/mul/div 指令        │
  │ arith.cmpi           │ offs < N (mask)          │ → GPU compare 指令            │
  │ arith.select         │ tl.where(cond, a, b)     │ → GPU conditional move        │
  │ scf.for              │ for k in range(...):     │ → 循环结构（后续可能被展开）     │
  │ scf.if               │ if cond: ...             │ → 条件分支                     │
  └──────────────────────┴──────────────────────────┴────────────────────────────────┘

  🔑 constexpr 参数的命运:
    你的 Python:  def kernel(..., BLOCK: tl.constexpr):
    在 TTIR 中:   BLOCK 不存在了！它已经被替换成具体的常数。
                  arith.muli %pid, %c64_i32  ← "64" 是编译进去的
                  tt.make_range {start=0, end=64}  ← 不叫 "BLOCK" 了

    这意味着: 每个不同的 BLOCK 值 → 不同的 TTIR → 不同的编译结果。
    这也是 triton.autotune 存在的理由——它为每个 config 生成不同的代码。""")

    print("\n📖 下一步: python phase4_compiler/03_to_ttgir.py")
    print("   看 TTIR 如何变成 TTGIR，layout encoding 如何出现。\n")


if __name__ == "__main__":
    main()
