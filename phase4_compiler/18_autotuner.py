"""
18_autotuner.py — Autotuner 内部机制

学习目标:
  1. 理解 Triton 的 autotune 搜索算法
  2. 看懂 autotune cache 的结构
  3. 学会分析 autotune 输出，判断是否搜索充分

运行: python phase4_compiler/18_autotuner.py

前提: 已完成 01-17。
"""

import os
import json
import time
from pathlib import Path

import torch
import triton
import triton.language as tl


# ══════════════════════════════════════════════════════════════════════
# 一个带 autotune 的 kernel
# ══════════════════════════════════════════════════════════════════════


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=16),
    ],
    key=["N"],
)
@triton.jit
def autotuned_vector_add(x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    """带 autotune 的 vector add。"""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


# ══════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════


def main():
    print("=" * 70)
    print("  18 — Triton Autotuner 内部机制")
    print("=" * 70)

    # ── Autotune 概念 ─────────────────────────────────────
    print("─" * 70)
    print("  1. 什么是 Autotune?")
    print("─" * 70)
    print("""
  ╔══════════════════════════════════════════════════════════════════╗
  ║  Autotune = 穷举搜索 最优编译参数                                ║
  ╚══════════════════════════════════════════════════════════════════╝

  Triton 的 autotuner 本质上是一个 brute-force 搜索器:
    1. 你提供一个 configs 列表
    2. 对于每组合适的 config:
       a. 编译 kernel (生成 PTX/CUBIN)
       b. 在 GPU 上运行
       c. 测量耗时
    3. 缓存最优 config
    4. 后续调用直接用最优 config

  这不是机器学习的"autotune" — 没有贝叶斯优化，没有遗传算法。
  它就是 systematic grid search + caching。""")

    # ── 关键参数 ────────────────────────────────────────
    print("─" * 70)
    print("  2. @triton.autotune 的关键参数")
    print("─" * 70)
    print("""
  @triton.autotune(
      configs=[...],      ← 搜索空间: 所有可能的配置组合
      key=['M','N','K'],  ← cache key: 哪些参数决定选哪个 config
      prune_configs_by={  ← (可选) 剪枝规则: 提前排除不合法 config
          'early_config_prune': my_prune_fn,
      },
      warmup=25,          ← 每个 config 预热多少次 (默认 25)
      rep=100,            ← 每个 config 测量多少次 (默认 100)
  )

  key 参数详解:
    key=['N']:
      → 不同的 N 值 → 不同的 cache entry → 可能选不同的 config
      → 例: N=1024 和 N=4096 各自独立 autotune

    key=['M','N','K']:
      → 不同的 (M,N,K) 组合 → 不同的 cache entry
      → 更精确，但 cache 条目更多

  prune_configs_by:
    → 在搜索前剪枝:
      如果 BLOCK_SIZE > N (block 比数据还大) → 剪掉这个 config
    → 节省编译时间

  warmup 和 rep:
    → warmup: GPU 预热次数 (让 clock 稳定)
    → rep: 实际测量次数 (取平均或最小)
    → 总运行次数 = len(configs) × (warmup + rep)
""")

    # ── Cache 结构 ───────────────────────────────────────
    print("─" * 70)
    print("  3. Autotune Cache 结构")
    print("─" * 70)

    cache = Path.home() / ".triton" / "cache"
    if cache.exists():
        dirs = sorted(cache.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        print(f"  Cache 根目录: {cache}")
        print(f"  子目录总数: {sum(1 for d in cache.iterdir() if d.is_dir())}")
        print(f"\n  最近的 5 个目录:")
        for d in dirs[:5]:
            if d.is_dir():
                entries = list(d.iterdir())
                has_ptx = any(e.suffix == '.ptx' for e in entries)
                has_json = any(e.suffix == '.json' for e in entries)
                print(f"    {d.name}/")
                print(f"      PTX: {has_ptx}, JSON: {has_json}, 文件数: {len(entries)}")
                for e in entries[:4]:
                    print(f"      {e.name} ({e.stat().st_size} bytes)")
                if len(entries) > 4:
                    print(f"      ... 及 {len(entries) - 4} 个文件")

    # ── 查看 autotune cache JSON ─────────────────────────
    print("\n" + "─" * 70)
    print("  4. Autotune Cache JSON 文件解析")
    print("─" * 70)

    # 找 autotune 相关的 JSON 文件
    json_files = list(cache.rglob("*.json")) if cache.exists() else []
    if json_files:
        latest_json = sorted(json_files, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        print(f"  最新 JSON: {latest_json.name}")
        try:
            data = json.loads(latest_json.read_text())
            # 只展示 key (不展示完整值，可能很长)
            if isinstance(data, dict):
                for k in list(data.keys())[:5]:
                    v = data[k]
                    if isinstance(v, str) and len(v) > 100:
                        v = v[:100] + "..."
                    print(f"    {k}: {v}")
        except Exception as e:
            print(f"    (无法解析: {e})")

    print("""
  Autotune cache 的典型结构:
    ~/.triton/cache/
    ├── <kernel_hash>/
    │   ├── <config_1_hash>/
    │   │   ├── *.ptx        ← 这个 config 编译出的 PTX
    │   │   └── *.cubin      ← 编译好的 CUDA 二进制
    │   ├── <config_2_hash>/
    │   │   └── ...
    │   └── __launcher_cache__/
    │       └── autotune.json ← 记录了哪个 config 最快以及耗时
""")

    # ── 运行 autotune ────────────────────────────────────
    print("─" * 70)
    print("  5. 观察 Autotune 执行过程")
    print("─" * 70)

    print("  运行 autotuned kernel (观察输出)...")
    os.environ["TRITON_PRINT_AUTOTUNING"] = "1"  # 打印 autotune 过程

    N = 1024
    x = torch.randn(N, device="cuda")
    y = torch.randn(N, device="cuda")
    out = torch.empty(N, device="cuda")

    start = time.time()
    autotuned_vector_add[(triton.cdiv(N, 128),)](x, y, out, N)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    print(f"\n  第一次调用耗时: {elapsed:.2f}s (包含 autotune 搜索)")
    print(f"  (第一次调用会比较慢，因为要搜索所有 config)")

    # 第二次调用 (从 cache 读取)
    start = time.time()
    autotuned_vector_add[(triton.cdiv(N, 128),)](x, y, out, N)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"  第二次调用耗时: {elapsed:.4f}s (从 cache 读取，应该很快)")

    # ── Autotune 最佳实践 ────────────────────────────────
    print("\n" + "─" * 70)
    print("  6. Autotune 最佳实践")
    print("─" * 70)
    print("""
  ✅ DO:
    • key 设置为影响性能的形状参数 (如 ['M','N','K'])
    • 提供足够多的 config (至少 5-10 个，覆盖不同 num_warps/block_size)
    • 使用 prune_configs_by 剪掉明显不好的 config
    • 第一次运行留足时间 (autotune 可能花几十秒)

  ❌ DON'T:
    • key 设为 num_warps 或 BLOCK_SIZE (这些是 config 参数，不是 cache key)
    • 提供 100+ 个 config (编译时间太长，边际收益递减)
    • 在生产环境中每次都 autotune (用 cache!)
    • 忘记设置 key (默认行为可能不准确)

  分析 autotune 结果:
    TRITON_PRINT_AUTOTUNING=1 python my_kernel.py
    → 打印每个 config 的测试耗时
    → 看哪个 config 胜出，哪些被剪掉

  如果 autotune 选了一个看起来不好的 config:
    → 检查是否 warmup 充分 (warmup 太小可能导致错误的测量)
    → 增加 rep 次数获得更稳定的测量
    → 可能是这个 config 确实最快 (寄存器使用刚好 fit SM 限制)
""")

    print("\n📖 下一步: python phase4_compiler/19_env_vars.py")
    print("   Triton 所有环境变量速查手册。\n")


if __name__ == "__main__":
    main()
