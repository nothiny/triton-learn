#!/usr/bin/env python3
"""
Phase 4 CuTe environment check and placeholder.

Detects whether CUTLASS/CuTe headers are available for C++ compilation.
CUTLASS is a header-only library, so no compiled libraries needed.
"""

import os
import sys
import subprocess
from pathlib import Path


def check_cutlass_available() -> bool:
    """Check if CUTLASS headers are findable."""
    # Common locations
    locations = [
        os.environ.get("CUTLASS_PATH", ""),
        "/usr/local/cutlass",
        Path.home() / "cutlass",
        Path.home() / "src" / "cutlass",
    ]

    for loc in locations:
        if not loc:
            continue
        include_dir = Path(loc) / "include"
        cute_header = include_dir / "cute" / "layout.hpp"
        if cute_header.exists():
            print(f"  CUTLASS found at: {loc}")
            print(f"  CuTe headers: {cute_header}")
            return True

    return False


def check_cuda_compiler() -> bool:
    """Check if nvcc is available for C++ GPU compilation."""
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            version_line = result.stdout.strip().split("\n")[-1]
            print(f"  nvcc found: {version_line}")
            return True
    except FileNotFoundError:
        pass
    except Exception:
        pass

    print("  nvcc not found — will need it for CUTLASS/CuTe")
    return False


def main():
    print("=" * 60)
    print("Phase 4: CuTe / CUTLASS — Environment Check")
    print("=" * 60)

    print("\n[1] CUTLASS:")
    has_cutlass = check_cutlass_available()
    if not has_cutlass:
        print("  CUTLASS not found.")
        print("  Install: git clone https://github.com/NVIDIA/cutlass.git")
        print("  Then set: export CUTLASS_PATH=/path/to/cutlass")

    print("\n[2] CUDA Compiler:")
    has_nvcc = check_cuda_compiler()

    print("\n[3] PyTorch + Triton (current stack):")
    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("  PyTorch not found")

    try:
        import triton
        print(f"  Triton: {triton.__version__}")
    except ImportError:
        print("  Triton not found")

    if has_cutlass and has_nvcc:
        print("\n✅ Ready to start CuTe learning!")
    else:
        print("\n⚠️  CUTLASS not fully set up — continue with Triton first.")
        print("   CuTe is Phase 4 material; Phase 1-3 are all Triton-based.")


if __name__ == "__main__":
    main()
