#!/usr/bin/env python3
"""
Phase 4 cuTile Python environment check and placeholder.

Detects whether NVIDIA cuTile Python (cuda-tile) is available.
cuTile is NVIDIA's official GPU kernel DSL, built on Tile IR.
"""

import sys


def check_cutile_available() -> bool:
    """Check if cuTile Python (cuda-tile) is installed."""
    try:
        import cuda.tile
        print(f"  cuTile Python found: {cuda.tile.__file__}")
        return True
    except ImportError:
        print("  cuTile Python not installed.")
        print("  Install: pip install cuda-tile[tileiras]")
        print("  Requires: CUDA Toolkit >= 13.1, NVIDIA Driver >= r580")
        return False


def check_cuda_version() -> bool:
    """Check CUDA version meets requirements."""
    try:
        import torch
        cuda_ver = torch.version.cuda
        major = int(cuda_ver.split(".")[0]) if cuda_ver else 0
        print(f"  PyTorch CUDA: {cuda_ver}")
        if major >= 13:
            print("  ✅ Meets cuTile requirement (>= 13.1)")
            return True
        else:
            print(f"  ⚠️  cuTile needs CUDA 13.1+, currently {cuda_ver}")
            return False
    except ImportError:
        print("  PyTorch not found — can't check CUDA version")
        return False


def main():
    print("=" * 60)
    print("Phase 4: cuTile Python — Environment Check")
    print("=" * 60)

    print("\n[1] cuTile Python (cuda-tile):")
    has_cutile = check_cutile_available()

    print("\n[2] CUDA Version:")
    has_cuda = check_cuda_version()

    print("\n[3] Current Stack (Triton):")
    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("  PyTorch not found")

    try:
        import triton
        print(f"  Triton: {triton.__version__}")
    except ImportError:
        print("  Triton not found")

    if has_cutile and has_cuda:
        print("\n✅ Ready to explore cuTile Python!")
        print("   Start: https://docs.nvidia.com/cuda/cutile-python")
        print("   Examples: https://github.com/NVIDIA/TileGym")
    else:
        print("\n⚠️  cuTile Python not set up — continue with Triton first.")
        print("   cuTile is Phase 4 material; Phase 1-3 are all Triton-based.")


if __name__ == "__main__":
    main()
