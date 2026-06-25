"""
01_dump_ir.py — Dump 各阶段 Triton IR

学习目标：
  - 学会设置环境变量 dump 各阶段 IR
  - 识别 TTIR / TTGIR / LLVM IR / PTX 的关键特征
  - 看 layout encoding 如何影响生成的代码

运行: python phase3_compiler/01_dump_ir.py
"""

import os
import sys
from pathlib import Path

# 启用 IR dump
os.environ["TRITON_KERNEL_DUMP"] = "1"
os.environ["TRITON_KERNEL_OVERRIDE"] = "1"  # 强制重新编译（跳过缓存）

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# 定义一个简单 kernel 用于 dump
# ---------------------------------------------------------------------------


@triton.jit
def simple_add_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    最简单的 vector add，用于观察 IR 管线。
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("01_dump_ir — Triton IR Pipeline")
    print("=" * 60)

    # 运行 kernel 以触发编译
    print("\n[1] Running kernel to trigger compilation...")
    size = 1024
    x = torch.rand(size, device="cuda")
    y = torch.rand(size, device="cuda")
    out = torch.empty_like(x)

    simple_add_kernel[(triton.cdiv(size, 1024),)](x, y, out, size, BLOCK_SIZE=1024)
    torch.cuda.synchronize()

    # 检查缓存目录
    print("\n[2] Checking Triton cache for dumped IR...")
    cache_dir = Path.home() / ".triton" / "cache"
    if cache_dir.exists():
        # 找到最近修改的文件
        files = sorted(cache_dir.rglob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
        print(f"  Cache directory: {cache_dir}")
        print(f"  Recent files ({min(10, len(files))}):")
        for f in files[:10]:
            if f.is_file():
                suffix = f.suffix
                label = {
                    ".ttir": "TTIR",
                    ".ttgir": "TTGIR",
                    ".ll": "LLVM IR",
                    ".ptx": "PTX",
                    ".cubin": "CUBIN",
                }.get(suffix, "")
                if label:
                    print(f"    [{label}] {f.name} ({f.stat().st_size} bytes)")

    # 通过 triton.compiler API 获取 IR（如果可用）
    print("\n[3] Attempting programmatic IR access...")
    try:
        from triton.compiler import compile as triton_compile
        print("  triton.compiler.compile is available")
        print("  (API details depend on Triton version)")
    except ImportError:
        print("  triton.compiler.compile not available (Triton < 3.0?)")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n[4] IR Dump Guide:")
    print("  --- TTIR (Triton IR, tt dialect) ---")
    print("  Key ops to look for:")
    print("    tt.load, tt.store    — memory access")
    print("    tt.dot               — matrix multiply (maps to MMA)")
    print("    tt.reduce            — reduction (sum, max, etc.)")
    print("    tt.broadcast         — broadcasting")
    print()
    print("  --- TTGIR (Triton GPU IR, tt + ttg dialect) ---")
    print("  Key additions:")
    print("    #blocked<{...}>      — BlockedEncodingAttr (thread→element mapping)")
    print("    #mma<{...}>          — MmaEncodingAttr (tensor core layout)")
    print("    ttg.convert_layout   — explicit layout conversion (may insert barriers)")
    print()
    print("  --- LLVM IR ---")
    print("  Standard LLVM IR for NVPTX target.")
    print("  Register allocation happens here by LLVM backend.")
    print()
    print("  --- PTX ---")
    print("  NVIDIA GPU assembly. Key instructions:")
    print("    ld.global / st.global   — HBM access")
    print("    ld.shared / st.shared   — shared memory access")
    print("    mma.sync.aligned        — Tensor Core MMA")
    print("    bar.sync                — barrier (sync within block)")
    print()
    print("  Environment variables for more detail:")
    print("    MLIR_PRINT_IR_AFTER_ALL=1    — dump after each MLIR pass")
    print("    TRITON_ALWAYS_COMPILE=1      — always recompile (no JIT cache)")


if __name__ == "__main__":
    main()
