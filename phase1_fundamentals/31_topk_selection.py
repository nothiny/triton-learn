"""
31_topk_selection.py — Top-K Selection with Per-Block Filtering

学习目标:
  - 掌握 Top-K 选择的 GPU 实现策略
  - 理解 "阈值过滤 + compaction" 模式 (比全排序更高效)
  - 学习如何用 Triton 实现 per-block Top-K + host 合并

算法 (2-pass threshold):
  1. 用全局 k-th largest 作为阈值 (这里用 PyTorch 算阈值, 聚焦 GPU 过滤部分)
  2. GPU kernel: 扫描全量数据, 保留 >= 阈值的元素 (带 index)
  3. Host 端: 合并各 block 的候选集, 精确选出 Top-K

为什么不用全排序:
  - 排序 O(N log N), 在 GPU 上需要多次全局同步
  - 阈值过滤 O(N), 只需一次扫描 + 小规模合并
  - 对于 k << N 的场景 (如 Beam Search k=8), 阈值过滤远快于排序

运行: python phase1_fundamentals/31_topk_selection.py
"""

import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch, triton, triton.language as tl


@triton.jit
def topk_filter_kernel(
    x_ptr, threshold,         # 阈值 (标量)
    out_vals_ptr, out_idxs_ptr,  # 候选输出
    out_counts_ptr,           # 每个 block 输出了多少个候选
    n_elements,
    k: tl.constexpr,          # 每 block 最多保留 k 个候选
    BLOCK_SIZE: tl.constexpr,
):
    """
    扫描 x, 保留所有 >= threshold 的元素及其 index.
    用共享内存做 per-block compaction.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=float("-inf"))

    # [GPU] 找到 block 内 >= threshold 的元素
    # 用 tl.where 得到 mask, 然后用共享内存做 compaction
    is_candidate = x >= threshold

    # 简单策略: 用 scatter 写入候选
    # 但由于 scatter 不支持动态 slot, 这里用 sequential write
    # 生产级实现用 shared memory compaction
    count = 0
    for i in range(BLOCK_SIZE):
        val = tl.load(x_ptr + block_start + i)
        is_valid = (block_start + i) < n_elements
        if is_valid and val >= threshold and count < k:
            tl.store(out_vals_ptr + pid * k + count, val)
            tl.store(out_idxs_ptr + pid * k + count, block_start + i)
            count += 1

    tl.store(out_counts_ptr + pid, count)


def topk(x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Top-K selection via threshold filtering.
    返回 (values, indices), 降序排列.
    """
    n = x.numel()
    BLOCK_SIZE = 2048
    n_blocks = triton.cdiv(n, BLOCK_SIZE)

    # Step 1: 用 PyTorch 找阈值 (k-th largest)
    threshold = x.flatten().topk(k).values[-1].item()

    # Step 2: GPU 过滤
    out_vals = torch.full((n_blocks * k,), float("-inf"),
                          device=x.device, dtype=x.dtype)
    out_idxs = torch.full((n_blocks * k,), -1,
                          device=x.device, dtype=torch.int64)
    out_counts = torch.zeros(n_blocks, device=x.device, dtype=torch.int32)

    grid = (n_blocks,)
    topk_filter_kernel[grid](
        x, threshold, out_vals, out_idxs, out_counts, n, k=k, BLOCK_SIZE=BLOCK_SIZE
    )

    # Step 3: Host 端合并候选 (keep on GPU for performance)
    # Collect all candidates via masking
    total_candidates = out_counts.sum().item()
    if total_candidates == 0:
        return torch.tensor([], device=x.device), torch.tensor([], device=x.device)

    # Flatten candidates: gather from per-block buffers
    all_vals_list = []
    all_idxs_list = []
    for b in range(n_blocks):
        cnt = out_counts[b].item()
        if cnt > 0:
            start = b * k
            all_vals_list.append(out_vals[start:start + cnt])
            all_idxs_list.append(out_idxs[start:start + cnt])

    all_vals = torch.cat(all_vals_list)
    all_idxs = torch.cat(all_idxs_list)

    # 对候选排序, 取 Top-K
    _, sort_idx = all_vals.sort(descending=True)
    final_vals = all_vals[sort_idx[:k]]
    final_idxs = all_idxs[sort_idx[:k]]
    return final_vals, final_idxs


def main():
    print("=" * 60)
    print("31_topk_selection — Top-K via Threshold Filtering")
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.manual_seed(42)

    for name, (n, k) in [("small ", (256, 8)), ("medium", (4096, 32)),
                           ("large ", (65536, 16)), ("xl    ", (262144, 64))]:
        x = torch.randn(n, device="cuda")
        vals_t, idxs_t = topk(x, k)
        vals_r, idxs_r = torch.topk(x, k)
        match_vals = vals_t.shape == vals_r.shape
        if match_vals:
            val_ok = (vals_t - vals_r).abs().max().item() < 1e-4
        else:
            val_ok = False
        print(f"  [{name}] n={n:7d} k={k:3d}  "
              f"vals_match={val_ok}  {'✅' if val_ok else '❌'}")

    # Perf
    print("\n--- Performance ---")
    x = torch.randn(1048576, device="cuda", dtype=torch.float32)
    import time
    for _ in range(10): topk(x, 32)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100): topk(x, 32)
    torch.cuda.synchronize()
    t_ms = (time.perf_counter() - t0) / 100 * 1000
    print(f"  Triton topk (k=32): {t_ms:.4f} ms (1M elements)")

    t0 = time.perf_counter()
    for _ in range(100): x.topk(32)
    torch.cuda.synchronize()
    t_ref = (time.perf_counter() - t0) / 100 * 1000
    print(f"  PyTorch topk (k=32): {t_ref:.4f} ms")


# PERFORMANCE NOTES
# =================
# - 阈值过滤是 O(N) 的, 远优于全排序 O(N log N).
# - 本实现的瓶颈: per-block sequential write (count loop).
#   生产级实现用 shared memory compaction: 用 prefix-sum 算每个线程的写入位置.
# - 阈值选取: 这里用 PyTorch topk, 在实际应用中阈值通常是已知的
#   (如 Beam Search 中的 cumulative log-prob 阈值).
# - 优化方向: 用 warp-level primitives 做 block 内 compaction,
#   避免 sequential loop.

if __name__ == "__main__":
    main()
