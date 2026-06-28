"""
04_layout_system.py — Layout Encoding 系统深度解析

学习目标:
  1. 全面理解 5 种 layout 类型及其适用场景
  2. 能用公式计算每个 layout 的总元素数
  3. 可视化 thread→element 的映射关系

运行: python phase4_compiler/04_layout_system.py

前提: 已运行 03，知道 layout encoding 在 TTGIR 中出现。
"""

# 这个文件不依赖 GPU（不需要编译 kernel），纯粹是概念讲解 + 可视化。
# 如果你想要看真实 layout，先运行 03_to_ttgir.py 再回来看这个。


def explain_blocked_encoding():
    """详细解释 BlockedEncodingAttr。"""
    print("─" * 70)
    print("  1. BlockedEncodingAttr — 标准分块布局")
    print("─" * 70)
    print("""
  这是最常见的 layout。几乎所有 elementwise op 的输出都用它。

  #blocked<{
    sizePerThread  = [s0, s1, ...],    ← 每个线程在每个维度的持有量
    threadsPerWarp = [t0, t1, ...],    ← warp 内线程在各维度的分布
    warpsPerCTA    = [w0, w1, ...],    ← CTA 内 warp 在各维度的分布
    order          = [o0, o1, ...]     ← 维度优先级（innermost first）
  }>

  核心公式:
    CTA 在各维度的总大小 = warpsPerCTA × threadsPerWarp × sizePerThread
    所有维度的乘积必须 ≥ tensor 的 shape

  例 1: 1D vector<128>
    #blocked<{
      sizePerThread  = [1],      ← 每个线程 1 个元素
      threadsPerWarp = [32],     ← 32 个线程沿 dim 0
      warpsPerCTA    = [4],      ← 4 个 warp
      order          = [0]       ← 只有 dim 0
    }>
    → 128 元素 = 4 warps × 32 threads/warp × 1 elem/thread ✓

  例 2: 2D tile<64, 128> (常见于 GEMM)
    #blocked<{
      sizePerThread  = [1, 4],   ← 每个线程 (1 elem in dim0, 4 elems in dim1)
      threadsPerWarp = [2, 16],  ← warp 内 2×16=32 线程
      warpsPerCTA    = [4, 1],   ← 4 warps 沿 dim0, 1 warp 沿 dim1
      order          = [0, 1]    ← dim0 innermost
    }>
    → dim0: 4 × 2 × 1 = 8  (但实际 64 行需要 64/wpc/tpw = 8 warps?)""")


def visualize_blocked_1d():
    """可视化 1D blocked layout 的线程映射。"""
    print("""
  ╔══════════════════════════════════════════════════════════════╗
  ║  可视化: 1D blocked layout                                  ║
  ║  tensor<128>, BLOCK=128, 4 warps × 32 threads × 1 elem      ║
  ╚══════════════════════════════════════════════════════════════╝

    Warp 0 (threads 0-31):           Warp 1 (threads 32-63):
    ┌──┬──┬──┬──┬──┬──┬──┬──┐      ┌──┬──┬──┬──┬──┬──┬──┬──┐
    │ 0│ 1│ 2│ 3│..│29│30│31│      │32│33│34│35│..│61│62│63│
    └──┴──┴──┴──┴──┴──┴──┴──┘      └──┴──┴──┴──┴──┴──┴──┴──┘
    ↑                              ↑
    thread 0 → elem 0              thread 0 → elem 32

    Warp 2 (threads 64-95):          Warp 3 (threads 96-127):
    ┌──┬──┬──┬──┬──┬──┬──┬──┐      ┌──┬──┬──┬──┬──┬──┬──┬──┐
    │64│65│66│67│..│93│94│95│      │96│97│98│99│..│125│126│127│
    └──┴──┴──┴──┴──┴──┴──┴──┘      └──┴──┴──┴──┴──┴──┴──┴──┘

    每个线程只处理 1 个元素 (sizePerThread=1)。
    线程 i 处理全局元素 i。
    这是最简单的情况——所有线程独立，不需要 shuffle 或 shared memory。""")


def visualize_blocked_2d():
    """可视化 2D blocked layout。"""
    print("""
  ╔══════════════════════════════════════════════════════════════╗
  ║  可视化: 2D blocked layout                                  ║
  ║  tensor<128×64>, 4 warps × 32 threads × (1×4) elems        ║
  ║  warpsPerCTA=[4,1], threadsPerWarp=[2,16], sizePerThread=[1,4] ║
  ╚══════════════════════════════════════════════════════════════╝

    CTA 的 128×64 tile 被分成 4 个 warp (沿 dim 0):

      dim 0 (128 rows)
      ┌──────────────────────────────────────────────────┐
      │ Warp 0: rows 0-31                                │
      │  ┌──────────────────────┐                        │
      │  │Thread (0,0): 1×4 elem │  ...  Thread (0,15)  │  ← threadsPerWarp=[2,16]
      │  │Thread (1,0): 1×4 elem │  ...                  │
      │  └──────────────────────┘                        │  ← 每个 thread 管 1×4=4 个元素
      ├──────────────────────────────────────────────────┤
      │ Warp 1: rows 32-63                               │
      ├──────────────────────────────────────────────────┤
      │ Warp 2: rows 64-95                               │
      ├──────────────────────────────────────────────────┤
      │ Warp 3: rows 96-127                              │
      └──────────────────────────────────────────────────┘

    每个 thread 持有:
      dim 0: 1 个元素  (sizePerThread[0]=1)
      dim 1: 4 个连续元素 (sizePerThread[1]=4, 对 coalescing 友好)

    注意 warp 内 thread 的排列: 2×16=32
      • threadsPerWarp[0]=2 → 2 "行"的 threads
      • threadsPerWarp[1]=16 → 每行 16 个 threads
      • 这是 Triton 编译器根据 BLOCK 大小和 num_warps 自动选择的""")


def explain_slice_encoding():
    """详细解释 SliceEncodingAttr。"""
    print("""
  ──────────────────────────────────────────────────────────────────
  2. SliceEncodingAttr — 规约操作的"切片"布局
  ──────────────────────────────────────────────────────────────────

  产生场景: 对 tensor 沿某个轴做 reduction (sum, max, min)

  例: 2D tensor → sum along axis=1 → 1D tensor
    输入: tensor<M×N, #blocked<{sizePerThread=[1,4], threadsPerWarp=[2,16], ...}>>
    输出: tensor<M, #slice<{dim=1, parent=#blocked<{...}>}>>

  SliceEncoding 的含义:
    把规约维度 (dim=1) 上的所有线程"折叠"起来。
    原本每个线程持有 dim=1 上的 4 个元素 → 规约后合为一个标量。
    这需要 warp shuffle (同 warp 内通信) + shared memory (跨 warp 通信)。

  关键: SliceEncoding 几乎总是需要一次 convert_layout 才能被后续 op 使用。
        这意味着 reduction → 后续使用之间有隐式的 shared memory round-trip。
""")


def explain_mma_encoding():
    """详细解释 MmaEncodingAttr。"""
    print("""
  ──────────────────────────────────────────────────────────────────
  3. MmaEncodingAttr — Tensor Core MMA 布局
  ──────────────────────────────────────────────────────────────────

  产生场景: tl.dot(a, b) 的输出

  #mma<{versionMajor=2, versionMinor=0, instrShape=[16,8,16]}>
    ↑                    ↑        ↑
    MMA 版本 (Ampere=2)   |        一个 MMA 指令处理的 shape (M×N×K)

  MmaEncodingAttr 描述的是 Tensor Core 需要的特殊数据排列。
  它不是"每线程 N 个元素"这么简单——
  Tensor Core 要求数据按 warp-level matrix layout 排列。

  H100 (SM90) 的 MMA shape 比 A100 (SM80) 更大:
    A100: m16n8k16  (16×8×16 per MMA instruction)
    H100: m16n8k32 (16×8×32 per MMA instruction, 更大的 K tile)

  MmaEncoding 的线程→元素映射由 Triton 编译器根据 GPU 架构决定，
  你不直接控制——通过 num_warps 和 BLOCK 大小间接影响。
""")


def explain_dot_operand_encoding():
    """详细解释 DotOperandEncodingAttr。"""
    print("""
  ──────────────────────────────────────────────────────────────────
  4. DotOperandEncodingAttr — tl.dot 的操作数布局
  ──────────────────────────────────────────────────────────────────

  产生场景: tl.dot(a, b) 的输入 (a, b)

  #dot_op<{opIdx=0, parent=#blocked<{...}>}>
    ↑        ↑            ↑
    A 操作数  opIdx=0 是 A     父布局 (数据原来的摆放方式)

  tl.dot(a, b) 的输入必须满足:
    • a 的 layout: opIdx=0, K 维是 innermost (A 侧)
    • b 的 layout: opIdx=1, K 维也是 innermost (B 侧)
    • 如果输入布局不对 → compiler 插入 convert_layout → 性能代价

  这是很多 Triton kernel 性能问题的根源:
    从 global memory 加载的数据默认是 #blocked<...>
    但 tl.dot 需要 #dot_op<...>
    如果编译器不能"智能地"把 load 直接放到正确的 layout，
    就会插入 convert_layout → 额外的 shared memory round-trip + barrier。
""")


def explain_scan_encoding():
    """详细解释 ScanEncodingAttr（较新，Triton 2.1+）。"""
    print("""
  ──────────────────────────────────────────────────────────────────
  5. ScanEncodingAttr — 前缀和/scan 操作的布局
  ──────────────────────────────────────────────────────────────────

  产生场景: tl.cumsum, tl.cumprod 等 scan 操作

  类似 SliceEncoding，但保留了 scan 维度上的依赖关系。
  Scan 需要同一 warp/block 内的线程之间通信。

  这个 encoding 在 Triton 2.1+ 中出现，目前还在发展中。
""")


# ══════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════


def main():
    print("=" * 70)
    print("  04 — Layout Encoding 系统深度解析")
    print("=" * 70)

    print("""
  Triton 编译器有 5 种 layout encoding，描述"数据如何分配到线程"。
  这是 Triton 区别于传统编译器最核心的设计。

  传统 GPU 编程 (CUDA):
    你手动计算 threadIdx.x → element index → 自己管理映射

  Triton:
    你写"对 tensor 做运算"，编译器自动分配 layout。
    你通过 BLOCK_SIZE, num_warps 等参数间接影响 layout 的选择。

  5 种 layout:
    1. BlockedEncodingAttr  — 标准分块 (90% 的 op 用这个)
    2. SliceEncodingAttr    — 规约后的切片
    3. MmaEncodingAttr      — Tensor Core MMA 输出
    4. DotOperandEncodingAttr — MMA 输入操作数
    5. ScanEncodingAttr     — Scan/前缀和

  每种 layout 决定了: 哪个线程持有 tensor 的哪些元素。""")

    # 逐一讲解
    explain_blocked_encoding()
    visualize_blocked_1d()
    visualize_blocked_2d()
    explain_slice_encoding()
    explain_mma_encoding()
    explain_dot_operand_encoding()
    explain_scan_encoding()

    # ── Layout 判断速查 ────────────────────────────────────
    print("─" * 70)
    print("  速查: 什么操作产生什么 layout?")
    print("─" * 70)
    print("""
  ┌─────────────────────┬────────────────────────┬──────────────────────┐
  │ Triton 代码           │ 产生的 layout            │ 说明                  │
  ├─────────────────────┼────────────────────────┼──────────────────────┤
  │ tl.load(ptr, mask)  │ #blocked<{...}>         │ 默认 blocked         │
  │ tl.store(ptr, val)  │ (不产生新 layout)        │ store 不改变 layout   │
  │ x + y               │ 继承输入的 layout        │ elementwise 不改变   │
  │ tl.sum(x, axis=1)   │ #slice<{dim=1, ...}>    │ 规约轴被"折叠"        │
  │ tl.dot(a, b)        │ #mma<{...}>             │ 输出用 MMA layout    │
  │ tl.dot 的输入 a     │ #dot_op<{opIdx=0, ...}> │ 如果可以原位转换       │
  │ tl.dot 的输入 b     │ #dot_op<{opIdx=1, ...}> │                      │
  │ convert_layout()    │ #blocked 或其他          │ 显式/隐式 layout 转换 │
  └─────────────────────┴────────────────────────┴──────────────────────┘""")

    # ── Layout 互转兼容性 ────────────────────────────────
    print("""
  ──────────────────────────────────────────────────────────────────
  Layout 互转兼容性
  ──────────────────────────────────────────────────────────────────

  不是所有 layout 都能互相转换。需要 convert_layout 时:
    - 有些转换可以直接做 (warp shuffle，同级寄存器交换)
    - 有些转换需要 shared memory round-trip (写入 shared memory → barrier → 读出来)
    - 最贵的转换: blocked ↔ mma，几乎总是需要 shared memory

  优化核心: 尽量减少 convert_layout 的次数。
  方式: 让数据一加载就以"正确"的 layout 存在。""")

    print("\n📖 下一步: python phase4_compiler/05_convert_layout.py")
    print("   看 layout conversion 如何成为隐形的性能杀手。\n")


if __name__ == "__main__":
    main()
