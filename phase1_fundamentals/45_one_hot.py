"""
45_one_hot.py — One-Hot Encoding (Scatter Pattern)

学习目标:
  - 掌握 one-hot scatter 的 GPU 实现: 根据 index 在指定位置写 1.0
  - 理解 scatter store 的随机写模式 (每个 thread 写到不同位置)
  - 学习 gather/scatter 的核心区别: load 不连续 vs store 不连续

数学定义:
  one_hot(x[i], num_classes) = vector of length num_classes
  where position x[i] = 1.0, all other positions = 0.0

GPU 上的挑战:
  - Scatter write: 每个 index 对应唯一 output 位置
  - 不像 gather (读随机位置), scatter (写随机位置) 可能引起 bank conflict
  - 但在 one-hot 场景, index 是正整数 (0..num_classes-1), 控制好 stride 即可

Gather vs Scatter:
  - Gather (27_embedding.py): output = weight[indices]  (input 连续, index 随机)
  - Scatter (45_one_hot): output[indices] = 1.0        (output 连续, index 随机)
  - Gather is "read random", Scatter is "write random"

为什么 One-Hot 在 GPU 上少用:
  - One-hot 会显著增加内存占用 (N → N × num_classes)
  - 生产中通常用 embedding lookup (sparse) 替代 one-hot (dense)
  - 但 one-hot 是理解 scatter 模式的最佳教学案例

运行: python phase1_fundamentals/45_one_hot.py
"""

import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch, triton, triton.language as tl
from utils.profiler import bench_compare, print_compare_report


@triton.jit
def one_hot_kernel(indices_ptr, output_ptr, n_elements, num_classes,
                    BLOCK_SIZE: tl.constexpr):
    """
    output[i, indices[i]] = 1.0, 其余位置 = 0.0.
    每个 program 处理一批输入 indices.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # [GPU] 读 indices
    idx = tl.load(indices_ptr + offsets, mask=mask, other=0)

    # [GPU] Scatter write: output[offsets * num_classes + idx] = 1.0
    # 每个 element 在 output 中的行起始 = offsets * num_classes
    # 加上 class index = idx → output_position = offsets * num_classes + idx
    row_starts = offsets * num_classes

    # 确保 idx 在有效范围
    valid = mask & (idx >= 0) & (idx < num_classes)
    out_offsets = row_starts + idx

    # [GPU] 原子写入 1.0 — 不同 thread 写不同位置, 无冲突
    tl.store(output_ptr + out_offsets, 1.0, mask=valid)


def one_hot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    """One-hot encode: output[i, indices[i]] = 1.0."""
    n = indices.numel()
    output = torch.zeros(n, num_classes, device=indices.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    one_hot_kernel[grid](indices, output, n, num_classes, BLOCK_SIZE=256)
    return output


def main():
    print("=" * 60)
    print("45_one_hot — One-Hot Encoding (Scatter Pattern)")
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.manual_seed(42)

    for name, (n, nc) in [("small ", (8, 5)), ("medium", (256, 32)),
                            ("large ", (4096, 128))]:
        indices = torch.randint(0, nc, (n,), device="cuda")
        y_t = one_hot(indices, nc)
        y_r = torch.nn.functional.one_hot(indices, nc).float()
        match = (y_t == y_r).all().item()
        print(f"  [{name}] n={n} classes={nc}  match={match}  "
              f"{'✅' if match else '❌'}")

    # Demo: compare size
    print("\n--- Memory Implication ---")
    n, nc = 1024, 50000  # e.g., vocab size for LLM
    indices = torch.randint(0, nc, (n,), device="cuda")
    print(f"  Indices size: {indices.numel() * 4 / 1024:.1f} KB ({n} int32)")
    print(f"  One-hot size: {n * nc * 4 / 1e9:.2f} GB ({n}x{nc} float32)")
    print(f"  💡 One-hot is {nc}x larger → embedding lookup preferred for large vocab")

    print("\n--- Performance ---")
    indices = torch.randint(0, 256, (65536,), device="cuda")
    n = indices.numel()
    result = bench_compare(
        {
            "Triton one_hot (scatter)": lambda: one_hot(indices, 256),
            "PyTorch F.one_hot": lambda: torch.nn.functional.one_hot(indices, 256).float(),
        },
        flops=0,               # 纯 scatter, 无计算
        bytes_accessed=n * 4 + n * 256 * 4,  # read indices + write one-hot
        dtype="fp32",
    )
    print_compare_report(result)


# PERFORMANCE NOTES
# =================
# - One-hot = scatter store: 每个 index 在 output 的唯一位置写入 1.0.
# - Scatter 的瓶颈: HBM 写入量 = N × num_classes × 4 bytes.
#   Embedding (gather) 只写 O(N × embed_dim), one-hot 写 O(N × num_classes).
#   num_classes (如 50000) >> embed_dim (如 768) → one-hot 浪费巨大.
# - 生产中 one-hot 几乎不用 (用 CrossEntropyLoss 直接接 logits).

if __name__ == "__main__":
    main()
