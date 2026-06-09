"""
04_ptx_analysis.py — 读取并注释生成的 PTX

学习目标：
  - 阅读 PTX（NVIDIA GPU 汇编）
  - 识别关键指令模式
  - 理解寄存器压力和 shared memory 使用

运行: python phase3_compiler/04_ptx_analysis.py
"""

import os
import torch
import triton
import triton.language as tl

from utils.ir_dump import annotate_ptx


@triton.jit
def analysis_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """A kernel with diverse operations for PTX analysis."""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Multiple operations to generate interesting PTX
    z = x + y
    w = z * 2.0 + 1.0  # FMA opportunity
    result = tl.maximum(w, 0.0)  # ReLU-like

    tl.store(out_ptr + offsets, result, mask=mask)


def main():
    print("=" * 60)
    print("04_ptx_analysis — PTX Code Analysis")
    print("=" * 60)

    # Enable dump
    os.environ["TRITON_KERNEL_DUMP"] = "1"

    # Run kernel
    size = 4096
    x = torch.rand(size, device="cuda")
    y = torch.rand(size, device="cuda")
    out = torch.empty_like(x)

    analysis_kernel[(triton.cdiv(size, 1024),)](x, y, out, size, BLOCK_SIZE=1024)
    torch.cuda.synchronize()

    print("\nPTX Analysis Guide:")
    print("  1. Look for .ptx files in ~/.triton/cache/")
    print("  2. Key instructions to identify:")
    print()
    print("  Memory hierarchy:")
    print("    ld.global.ca.f32   — cached global load (via L1)")
    print("    ld.global.cg.f32   — global load bypassing L1 (cache-global)")
    print("    st.global.f32      — global store")
    print("    ld.shared.f32      — shared memory load")
    print("    st.shared.f32      — shared memory store")
    print()
    print("  Compute:")
    print("    add.f32            — float add")
    print("    fma.rn.f32         — fused multiply-add (FMA)")
    print("    max.f32            — max (used for ReLU, etc.)")
    print("    mma.sync.aligned   — Tensor Core MMA")
    print()
    print("  Synchronization:")
    print("    bar.sync 0         — block-level barrier")
    print("    bar.warp.sync      — warp-level barrier")
    print("    membar.cta          — CTA memory fence")
    print()
    print("  Register analysis:")
    print("    .reg .f32 %r<N>    — float register declaration")
    print("    Count .reg declarations → estimate register pressure")
    print("    H100: max 255 regs/thread (but occupancy drops at >128)")
    print()
    print("  Shared memory:")
    print("    .shared .align 16 .b8 ...  — shared memory allocation")

    # Try to find and annotate PTX
    from pathlib import Path
    cache_dir = Path.home() / ".triton" / "cache"
    ptx_files = sorted(cache_dir.rglob("*.ptx"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if ptx_files:
        print(f"\nFound {len(ptx_files)} PTX file(s). Most recent:")
        latest = ptx_files[0]
        print(f"  {latest}")
        ptx_src = latest.read_text()
        annotated = annotate_ptx(ptx_src)
        # Print first 80 annotated lines
        print("\nAnnotated PTX (first 80 lines):")
        for line in annotated[:80]:
            print(line)
    else:
        print("\nNo PTX files found in cache.")


if __name__ == "__main__":
    main()
