"""
24_triton_tt_dialect.py — Triton 的 `tt` Dialect 完整参考

学习目标:
  1. 理解 tt dialect 的每一个 op: 语义、operand、result、约束
  2. 建立 "Python tl.* → TTIR op" 的精确映射
  3. 知道每个 op 在 compiler pass 中如何被 lowering

运行: python phase4_compiler/24_triton_tt_dialect.py

前提: 已完成 22-23。
"""

import os
from pathlib import Path

os.environ["TRITON_KERNEL_DUMP"] = "1"
os.environ["TRITON_KERNEL_OVERRIDE"] = "1"

import torch
import triton
import triton.language as tl


# ══════════════════════════════════════════════════════════════════════
# 生成包含多种 tt op 的 IR
# ══════════════════════════════════════════════════════════════════════


@triton.jit
def tt_op_catalog(x_ptr, y_ptr, out_ptr, N,
                  BLOCK: tl.constexpr,
                  USE_REDUCE: tl.constexpr,
                  USE_BROADCAST: tl.constexpr):
    """
    一个"op 目录" kernel，包含 tt dialect 的大多数 op。
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)     # make_range
    mask = offs < N                               # cmpi → splat → addi

    x = tl.load(x_ptr + offs, mask=mask)           # load
    y = tl.load(y_ptr + offs, mask=mask)

    z = x + y * 2.0                               # addf, mulf

    if USE_REDUCE:
        s = tl.sum(z)                              # reduce (sum)
        z = z + s                                  # broadcast (隐式)

    if USE_BROADCAST:
        z = tl.broadcast_to(z, (BLOCK,))           # 显式 broadcast

    tl.store(out_ptr + offs, z, mask=mask)         # store


def get_fresh_ttir():
    cache = Path.home() / ".triton" / "cache"
    files = sorted(cache.rglob("*.ttir"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0].read_text() if files else None


# ══════════════════════════════════════════════════════════════════════
# tt dialect 完整参考 (数据来自 Triton 源码和实际 IR 分析)
# ══════════════════════════════════════════════════════════════════════

TT_DIALECT_REFERENCE = {
    # ── Memory Operations ──
    "Memory": [
        {
            "op": "tt.load",
            "python": "tl.load(ptr, mask=..., other=..., cache_modifier=...)",
            "operands": "ptr: tensor<Nx!tt.ptr<T>>, mask: tensor<Nxi1>",
            "result": "tensor<NxT>",
            "attrs": "{CacheModifier = .ca/.cg/.cs, IsVolatile = false}",
            "lowering": "TTIR → TTGIR: same op, type 加上 #blocked layout\n"
                        "TTGIR → LLVM: 展开为 getelementptr + load + 条件 select (mask)",
            "notes": "最重要的 load op。mask 决定哪些元素实际加载 (越界时为 other 值)。"
        },
        {
            "op": "tt.store",
            "python": "tl.store(ptr, value, mask=...)",
            "operands": "ptr: tensor<Nx!tt.ptr<T>>, value: tensor<NxT>, mask: tensor<Nxi1>",
            "result": "无 (side-effect op)",
            "attrs": "{CacheModifier = .wb/.cg/.cs/.wt}",
            "lowering": "TTIR → TTGIR: 加 layout\n"
                        "TTGIR → LLVM: 展开为 getelementptr + 条件 store",
            "notes": "无返回值 (void op)。mask 为 false 的 element 不写入。"
        },
    ],

    # ── Data Movement ──
    "Data Movement": [
        {
            "op": "tt.splat",
            "python": "(隐式, 如 scalar + tensor)",
            "operands": "scalar: T",
            "result": "tensor<NxT> (或 pointer 版本)",
            "attrs": "无",
            "lowering": "展开为 broadcast (每个线程复制标量)",
            "notes": "把标量复制到 tensor 的每个元素。\n"
                     "Python 中 scalar + tensor 时编译器自动插入。"
        },
        {
            "op": "tt.broadcast",
            "python": "tl.broadcast_to(x, shape)",
            "operands": "x: tensor<SxT>",
            "result": "tensor<NxT> (N >= S)",
            "attrs": "无",
            "lowering": "展开为跨线程/跨 warp 的数据复制",
            "notes": "显式广播。常用于规约后恢复 shape。"
        },
        {
            "op": "tt.trans",
            "python": "(无直接 API, 编译器可能自动插入)",
            "operands": "x: tensor<MxNxT>",
            "result": "tensor<NxMxT>",
            "attrs": "{order = [1, 0]}",
            "lowering": "可能触发 layout 转换",
            "notes": "转置操作。在 TTIR 中较少出现，常被 layout change 替代。"
        },
    ],

    # ── Pointer Arithmetic ──
    "Pointer Arithmetic": [
        {
            "op": "tt.addptr",
            "python": "(隐式, 如 ptr + offsets)",
            "operands": "ptr: tensor<Nx!tt.ptr<T>>, offset: tensor<Nxi32>",
            "result": "tensor<Nx!tt.ptr<T>>",
            "attrs": "无",
            "lowering": "TTGIR → LLVM: getelementptr",
            "notes": "指针算术。Python 中的 ptr + offs 被翻译为此 op。"
        },
    ],

    # ── Index / Range ──
    "Index & Range": [
        {
            "op": "tt.get_program_id",
            "python": "tl.program_id(axis)",
            "operands": "axis: x/y/z",
            "result": "i32",
            "attrs": "{axis = 0/1/2 : i32}",
            "lowering": "TTGIR → LLVM: nvvm.read.ptx.sreg.ctaid.x",
            "notes": "获取 block 索引 (grid 中的位置)。axis 是 attribute 不是 operand！"
        },
        {
            "op": "tt.get_num_programs",
            "python": "tl.num_programs(axis)",
            "operands": "axis: x/y/z",
            "result": "i32",
            "attrs": "{axis = 0/1/2 : i32}",
            "lowering": "TTGIR → LLVM: nvvm.read.ptx.sreg.nctaid.x",
            "notes": "获取 grid 大小。"
        },
        {
            "op": "tt.make_range",
            "python": "tl.arange(start, end)",
            "operands": "无",
            "result": "tensor<Nxi32>",
            "attrs": "{start = 0 : i32, end = N : i32}",
            "lowering": "TTGIR → LLVM: tid.x + block_start (展开为线程索引)",
            "notes": "生成 [start, start+1, ..., end-1] 的 tensor。\n"
                     "start 和 end 必须是 attribute (编译期常量)！"
        },
    ],

    # ── Compute ──
    "Compute": [
        {
            "op": "tt.dot",
            "python": "tl.dot(a, b, allow_tf32=False)",
            "operands": "a: tensor<MxKxT>, b: tensor<KxNxT>",
            "result": "tensor<MxNxT_out> (通常 f32)",
            "attrs": "{allow_tf32 = false, max_num_imprecise_acc = ...}",
            "lowering": "TTIR → TTGIR: operands 变成 #dot_op layout, result 变成 #mma layout\n"
                        "AccelerateMatmul: 替换为 MMA op sequence",
            "notes": "★★★ Triton 最重要的 op。是编译器识别「矩阵乘」的关键标记。"
        },
        {
            "op": "tt.reduce",
            "python": "tl.sum(x, axis), tl.max(x, axis), tl.argmax(x, axis)",
            "operands": "x: tensor<SxT>, axis: i32",
            "result": "tensor<S'xT> (reduce 了一维)",
            "attrs": "{axis = 0/1/... : i32}",
            "lowering": "TTGIR: 加入 SliceEncoding → warp shuffle + shared memory",
            "notes": "规约操作。axis 在 attribute 中指定。\n"
                     "不同的规约类型 (sum/max/argmax) 对应不同的 arith op。"
        },
        {
            "op": "arith.addf / mulf / divf",
            "python": "x + y, x * y, x / y",
            "operands": "x: T, y: T",
            "result": "T",
            "attrs": "无",
            "lowering": "直接变为 PTX 算术指令",
            "notes": "来自 arith dialect，不是 tt dialect。\n"
                     "但和 tt op 混用 — 这就是 MLIR 的 multi-dialect 特性。"
        },
        {
            "op": "arith.cmpi",
            "python": "offs < N (生成 mask)",
            "operands": "lhs: tensor<Nxi32>, rhs: tensor<Nxi32>",
            "result": "tensor<Nxi1> (i1 = boolean)",
            "attrs": "{predicate = slt/sgt/sle/sge/eq/ne}",
            "lowering": "TTGIR → LLVM: icmp → PTX: setp",
            "notes": "整数比较，生成 mask tensor。\n"
                     "predicate attribute 决定比较类型。"
        },
    ],

    # ── Control Flow ──
    "Control Flow": [
        {
            "op": "scf.for",
            "python": "for i in range(start, end, step):",
            "operands": "loop 变量 (迭代计数器 + 循环携带值)",
            "result": "循环携带值的最终值",
            "attrs": "{lowerBound, upperBound, step}",
            "lowering": "可能被展开 (TritonGPUPipeline) 或保留为 LLVM loop",
            "notes": "来自 scf dialect。Pipeline pass 可能将其展开为 async copy。"
        },
        {
            "op": "scf.if",
            "python": "if tl.constexpr condition: ... else: ...",
            "operands": "condition: i1",
            "result": "两个分支的结果 (类型必须一致)",
            "attrs": "无",
            "lowering": "TTGIR → LLVM: br/cond_br → PTX: bra",
            "notes": "条件分支。不同于 Python if (编译期条件)，这是运行时分支。"
        },
    ],
}


# ══════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════


def main():
    print("=" * 70)
    print("  24 — Triton `tt` Dialect 完整参考")
    print("=" * 70)

    # 生成 IR
    N = 256
    x = torch.randn(N, device="cuda")
    y = torch.randn(N, device="cuda")
    out = torch.empty(N, device="cuda")
    tt_op_catalog[(1,)](x, y, out, N, BLOCK=256,
                          USE_REDUCE=True, USE_BROADCAST=True)
    torch.cuda.synchronize()

    ttir_text = get_fresh_ttir()
    found_ops = set()
    if ttir_text:
        import re
        for m in re.finditer(r'(\w+)\.(\w+)', ttir_text):
            found_ops.add(f"{m.group(1)}.{m.group(2)}")

    # ── 完整参考 ────────────────────────────────────────
    for category, ops in TT_DIALECT_REFERENCE.items():
        print(f"\n  {'═' * 65}")
        print(f"  {category}")
        print(f"  {'═' * 65}")
        for op_info in ops:
            op_name = op_info["op"]
            # 检查是否在生成的 IR 中出现过
            short_name = op_name.split(".")[-1]
            in_ir = op_name in found_ops
            status = "✅" if in_ir else "  "
            print(f"""
  {status} {op_name}
      Python:  {op_info['python']}
      Operand: {op_info['operands']}
      Result:  {op_info['result']}
      Attrs:   {op_info['attrs']}
      Lowering:
        {op_info['lowering'].strip()}
      💡 {op_info['notes'].strip()}
""")

    # ── 真实 IR 中出现的 op 统计 ────────────────────────────
    print(f"  {'═' * 65}")
    print(f"  真实 TTIR 中检测到的所有 dialect.op")
    print(f"  {'═' * 65}")

    if ttir_text:
        from collections import Counter
        op_counts = Counter()
        for m in __import__('re').finditer(r'(\w+)\.(\w+)', ttir_text):
            full = f"{m.group(1)}.{m.group(2)}"
            # 排除一些非 op 的匹配
            if not any(skip in full for skip in ["param.", "reg.", "global."]):
                op_counts[full] += 1

        for op, count in op_counts.most_common(20):
            print(f"    {op:35s} {count}x")

    # ── Python → TTIR 映射速查 ──────────────────────────
    print(f"""
  {'═' * 65}
  Python → TTIR 映射速查
  {'═' * 65}

  Python 代码                           TTIR op
  ─────────────────────────────────────────────────────────────
  tl.load(ptr + offs, mask=mask)   →   tt.splat (ptr) + tt.addptr + tt.load
  tl.store(ptr + offs, val, mask)  →   tt.splat (ptr) + tt.addptr + tt.store
  tl.arange(0, BLOCK)              →   tt.make_range {{start=0, end=BLOCK}}
  tl.program_id(0)                 →   tt.get_program_id x
  tl.program_id(1)                 →   tt.get_program_id y
  tl.dot(a, b)                     →   tt.dot (★★★)
  tl.sum(x, axis=1)                →   tt.reduce {{axis=1}}
  tl.max(x, axis=0)                →   tt.reduce {{axis=0}} + arith.maxf
  tl.zeros(shape, dtype)           →   arith.constant 0 + tt.splat
  x + y                            →   arith.addf
  x * y                            →   arith.mulf
  x / y                            →   arith.divf
  offs < N                         →   tt.splat(N) + arith.cmpi slt
  tl.broadcast_to(x, shape)        →   tt.broadcast
  pid * BLOCK + offs               →   arith.muli + arith.addi
  for k in range(0, K, BLOCK_K):   →   scf.for
  if tl.constexpr flag:            →   编译期消除 (不产生 op)
  if condition:                    →   scf.if

  🔑 关键:
    • 指针运算: Python 的 ptr + offset → splat + addptr 两个 op
    • Mask: Python 的 offs < N → splat + cmpi 两个 op
    • 简单的 Python 表达式变成了多个 MLIR op — 这是正常的！
    • 编译器之后会优化: 合并 splat, 消除冗余 addptr, 等等
""")

    print("\n📖 下一步: python phase4_compiler/25_triton_ttg_dialect.py")
    print("   Triton 的 ttg dialect — layout encoding 和异步操作。\n")


if __name__ == "__main__":
    main()
