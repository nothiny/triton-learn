"""
09_pipeline_prefetch.py — Software Pipelining & Prefetch

学习目标:
  1. 理解 num_stages 参数如何影响生成的代码
  2. 看懂 cp.async + commit_group + wait_group 的流水线模式
  3. 知道如何利用 pipelining 隐藏内存延迟

运行: python phase4_compiler/09_pipeline_prefetch.py

前提: 已运行 08，理解 pass pipeline 的基本结构。
"""

import os
from pathlib import Path

os.environ["TRITON_KERNEL_DUMP"] = "1"
os.environ["TRITON_KERNEL_OVERRIDE"] = "1"

import torch
import triton
import triton.language as tl


# ══════════════════════════════════════════════════════════════════════
# 一个包含循环的 kernel，用于观察 pipelining
# ══════════════════════════════════════════════════════════════════════


@triton.jit
def matmul_with_stages(A, B, C,
                        M, N, K,
                        BLOCK_M: tl.constexpr,
                        BLOCK_N: tl.constexpr,
                        BLOCK_K: tl.constexpr,
                        NUM_STAGES: tl.constexpr):
    """
    带 num_stages 参数的 matmul tile。
    这个 kernel 不直接用 autotune，而是让我们手工观察不同 num_stages 的效果。

    NUM_STAGES 控制 software pipelining 的深度:
      = 1: 无 pipelining (串行 load→compute→load→compute)
      = 2: 双缓冲 (load 下一块时计算当前块)
      = 3+: 更多缓冲 (更高吞吐，但需要更多 shared memory)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    A_ptr = A + rm[:, None] * K + rk[None, :]
    B_ptr = B + rk[:, None] * N + rn[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 这个循环是 pipelining 的目标
    for k in range(0, K, BLOCK_K):
        a = tl.load(A_ptr + k)
        b = tl.load(B_ptr + k * N)
        acc += tl.dot(a, b)

    tl.store(C + rm[:, None] * N + rn[None, :], acc)


# ══════════════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════════════


def find_latest_ptx():
    cache = Path.home() / ".triton" / "cache"
    if not cache.exists():
        return None
    files = sorted(cache.rglob("*.ptx"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def find_latest_ttgir():
    cache = Path.home() / ".triton" / "cache"
    if not cache.exists():
        return None
    files = sorted(cache.rglob("*.ttgir"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def check_async_copy(ptx_source: str) -> list[str]:
    """检查 PTX 中是否有异步拷贝指令 (pipelining 的标志)。"""
    lines = []
    for line in ptx_source.split("\n"):
        if "cp.async" in line or "commit_group" in line or "wait_group" in line:
            lines.append(line.strip())
    return lines


# ══════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════


def main():
    print("=" * 70)
    print("  09 — Software Pipelining & Prefetch")
    print("=" * 70)

    print("""
  ╔══════════════════════════════════════════════════════════════════╗
  ║  GPU 的内存延迟很高 (HBM ~300-800 cycles)。                     ║
  ║  Software pipelining 让计算和加载在时间上重叠，隐藏这个延迟。      ║
  ╚══════════════════════════════════════════════════════════════════╝

  概念类比: 餐厅厨房
    num_stages=1 (无 pipeline): 厨师(计算)等服务员(加载)端来下一道菜才开始做
    num_stages=2 (双缓冲):     服务员端第2道菜时，厨师正在做第1道菜
    num_stages=3:              第3道菜在路上时，厨师做第1道，服务员端第2道

  Triton 实现:
    1. 分析循环结构 (for k in range(0, K, BLOCK_K))
    2. 确定 initiation interval (多久启动一次新的"加载")
    3. 展开循环 → 插入异步拷贝指令 (cp.async)
    4. 插入 commit_group / wait_group 来管理依赖关系""")

    # ── 核心概念图解 ──────────────────────────────────────
    print("""
  ──────────────────────────────────────────────────────────────────
  核心概念: num_stages 怎么改变指令调度
  ──────────────────────────────────────────────────────────────────

  num_stages=1 (无 pipelining):
    时间 →
    |── load[0] ──|── compute[0] ──|── load[1] ──|── compute[1] ──|

  num_stages=2 (双缓冲):
    时间 →
    |── load[0] ──|── load[1] ──|── load[2] ──|
                    |── compute[0] ──|── compute[1] ──|
    (加载和计算在时间上重叠 → 总耗时减少)

  num_stages=3:
    时间 →
    |── load[0] ──|── load[1] ──|── load[2] ──|── load[3] ──|
                  |              |── compute[0] ──|── compute[1] ──|
    (更多的重叠 → 更高的吞吐，但需要更多 shared memory)

  🔑 关键权衡:
    num_stages 越大:
      ✅ 更少的内存等待 (加载和计算重叠更多)
      ❌ 更多的 shared memory (每个 stage 需要一个 buffer)
      ❌ 可能降低 occupancy (shared memory 用多了)
    → 最优值通常在 2-4 之间，由 autotune 自动确定""")

    # ── 实际观察: 运行不同 num_stages 的 kernel ──────────────
    print("─" * 70)
    print("  实际观察: 对比不同 num_stages 的 PTX")
    print("─" * 70)

    M, N, K = 256, 256, 256
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)

    for num_stages in [1, 2, 4]:
        print(f"\n  ▸ num_stages={num_stages}")
        C = torch.empty(M, N, device="cuda", dtype=torch.float32)
        matmul_with_stages[(1, 1)](A, B, C, M, N, K,
                                    BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
                                    NUM_STAGES=num_stages)
        torch.cuda.synchronize()

        ptx = find_latest_ptx()
        if ptx:
            content = ptx.read_text()
            async_ops = check_async_copy(content)
            print(f"     PTX: {ptx.name}")
            print(f"     cp.async / wait_group 指令: {len(async_ops)} 条")
            if async_ops:
                # 打印前几条
                for op in async_ops[:3]:
                    print(f"       {op[:100]}")
                if len(async_ops) > 3:
                    print(f"       ... 及 {len(async_ops) - 3} 条")
            else:
                # num_stages=1 不应该有 cp.async
                print(f"     (无异步拷贝 — num_stages=1 时正常)")

    # ── cp.async 工作机制 ─────────────────────────────────
    print("""
  ──────────────────────────────────────────────────────────────────
  cp.async 的工作机制 (以 num_stages=2 为例)
  ──────────────────────────────────────────────────────────────────

  编译器把原始的串行循环:
    for k in range(0, K, BLOCK_K):
        a = tl.load(A_ptr + k)       ← 同步加载 (线程等待数据)
        b = tl.load(B_ptr + k * N)
        acc += tl.dot(a, b)

  转换成 (简化版):
    // Stage 0: 预加载第一批数据
    cp.async.ca.shared.global [shared_buf_0], [A_ptr + 0]     // 异步！线程不等待
    cp.async.ca.shared.global [shared_buf_0], [B_ptr + 0]
    cp.async.commit_group                                        // 标记这组异步操作为 "group 0"

    for k in range(BLOCK_K, K, BLOCK_K):
        // 等待上一组完成
        cp.async.wait_group 0                                    // 等 group N-1 完成
        // 计算上一组
        a = ld.shared [shared_buf_0]                             // 从 shared memory 加载
        b = ld.shared [shared_buf_0]
        acc += mma(a, b)

        // 同时预取下一组
        cp.async.ca.shared.global [shared_buf_1], [A_ptr + k]   // 异步！不阻塞
        cp.async.ca.shared.global [shared_buf_1], [B_ptr + k]
        cp.async.commit_group

        // 切换 buffer (ping-pong between shared_buf_0 and shared_buf_1)

    // 最后一组
    cp.async.wait_group 0
    a = ld.shared [shared_buf_0/1]
    b = ld.shared [shared_buf_0/1]
    acc += mma(a, b)

  🔑 关键指令:
    cp.async.ca.shared.global [dst], [src]
      — 异步从 HBM 拷贝到 shared memory (CA = Cache All, 经过 L1)
      — 线程发起拷贝后立即返回，不等待数据到达

    cp.async.commit_group
      — 把前面发起的异步拷贝合并为一个 "group"
      — 之后可以用 wait_group 来等待这个 group 完成

    cp.async.wait_group N
      — 等待最近 N 个 group 中最老的那个完成
      — wait_group 0 = 等待所有已提交的异步拷贝完成

  💡 这也是为什么 num_stages 增加 → shared memory 增加:
    每个 stage 需要一个 shared memory buffer。
    num_stages=2 → 2 个 buffer (双缓冲)
    num_stages=3 → 3 个 buffer
    ...
    每个 buffer 的大小 = BLOCK_M × BLOCK_K × sizeof(dtype) (A) + 类似的 (B)""")

    # ── TTGIR 中的 pipeline ──────────────────────────────
    print("─" * 70)
    print("  TTGIR 中的 pipelining (概念性)")
    print("─" * 70)

    # 看看 TTGIR 中是否有 async_copy
    ttgir = find_latest_ttgir()
    if ttgir:
        content = ttgir.read_text()
        has_async = "async_copy" in content
        print(f"  TTGIR 中有 async_copy op: {has_async}")
        print(f"  (TritonGPUPipeline pass 在 TTGIR 中插入 async_copy)")
        # 找 async_copy 相关的行
        for line in content.split("\n"):
            if "async_copy" in line.lower() or "pipeline" in line.lower():
                print(f"    {line.strip()[:120]}")
                break

    # ── 实用建议 ────────────────────────────────────────
    print("""
  ──────────────────────────────────────────────────────────────────
  实用建议: num_stages 怎么调?
  ──────────────────────────────────────────────────────────────────

  1. 大多数情况: 使用 autotune (让它自动搜索)
     @triton.autotune(configs=[...], key=['M','N','K'])
     → Triton 会测试多个 num_stages 值，选最快的

  2. 手工调整的起点:
     计算密集 kernel (如大 tile matmul): num_stages=2 通常足够
     内存密集 kernel (如 elementwise): num_stages 没用 — 没有循环

  3. 判断是否太高的标志:
     • PTX 中 shared memory 声明很大 → 降低 num_stages
     • occupancy 下降 → 降低 num_stages
     • 没有性能提升 → 不需要更多 stage

  4. 不需要 pipelining 的情况:
     • 没有循环的 kernel (pipelining 只能优化循环)
     • 计算时间 << 加载时间 → pipelining 带来的重叠很小
     • shared memory 已经不够的情况""")

    print("\n📖 下一步: python phase4_compiler/10_register_pressure.py")
    print("   理解寄存器分配和三资源约束。\n")


if __name__ == "__main__":
    main()
