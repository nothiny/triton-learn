"""
30_argmax_reduce.py — Argmax with Index Tracking in Reduction

学习目标:
  - 掌握 "带索引的 reduction" 模式: 在求 max 的同时记录 argmax
  - 理解如何用 tl.argmax 或手动 (value, index) 对做比较
  - 为 Top-K 和 Attention 中的 argmax 操作打基础

GPU 执行特征:
  - tl.argmax 在 block 内用 warp shuffle 比较 (value, index) 对
  - 跨 block 用 atomic_max 或 store-then-reduce-on-host

运行: python phase1_fundamentals/30_argmax_reduce.py
"""

import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch, triton, triton.language as tl


@triton.jit
def argmax_kernel(x_ptr, val_out_ptr, idx_out_ptr, n_elements,
                   BLOCK_SIZE: tl.constexpr):
    """
    每个 program 处理一段, 找出段内 max value 和对应 index.
    Python wrapper 再合并各段的 (value, index) 得到全局 argmax.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=float("-inf"))

    # tl.argmax 返回 max value 和 index within block
    # [GPU] Triton lowers argmax to multi-phase warp shuffle
    val, idx = tl.max(x, axis=0, return_indices=True)

    # 全局 index = block_start + local_index
    global_idx = block_start + idx

    # [GPU] 标量写入 — 每个 block 只写 2 个标量
    tl.store(val_out_ptr + pid, val)
    tl.store(idx_out_ptr + pid, global_idx.to(tl.int32))


def argmax(x: torch.Tensor) -> tuple[int, float]:
    """
    返回 (argmax_index, max_value).
    分两步: (1) GPU 分段找 max, (2) host 端合并.
    """
    n = x.numel()
    BLOCK_SIZE = 1024
    n_blocks = triton.cdiv(n, BLOCK_SIZE)

    vals = torch.empty(n_blocks, device=x.device, dtype=torch.float32)
    idxs = torch.empty(n_blocks, device=x.device, dtype=torch.int32)

    grid = (n_blocks,)
    argmax_kernel[grid](x, vals, idxs, n, BLOCK_SIZE=BLOCK_SIZE)

    # Host 端选出各段中最大的
    best_block = vals.argmax().item()
    best_idx = idxs[best_block].item()
    best_val = vals[best_block].item()
    return best_idx, best_val


def main():
    print("=" * 60)
    print("30_argmax_reduce — Argmax with Index Tracking")
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.manual_seed(42)

    for name, size in [("small  ", 256), ("medium ", 4096), ("large  ", 65536)]:
        x = torch.randn(size, device="cuda")
        idx_t, val_t = argmax(x)
        val_r = x.max().item()
        idx_r = x.argmax().item()
        ok = "✅" if idx_t == idx_r and abs(val_t - val_r) < 1e-5 else "❌"
        print(f"  [{name}] size={size:6d}  "
              f"idx={idx_t:5d}/{idx_r:5d}  val={val_t:8.4f}/{val_r:8.4f}  {ok}")

    # Perf
    print("\n--- Performance ---")
    x = torch.randn(16777216, device="cuda", dtype=torch.float32)
    import time
    for _ in range(10): argmax(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100): argmax(x)
    torch.cuda.synchronize()
    t_ms = (time.perf_counter() - t0) / 100 * 1000
    bw = (x.numel() * 4) / (t_ms / 1000) / 1e9
    print(f"  Triton argmax: {t_ms:.4f} ms, {bw:.1f} GB/s (16M fp32)")

    t0 = time.perf_counter()
    for _ in range(100): x.argmax()
    torch.cuda.synchronize()
    t_ref = (time.perf_counter() - t0) / 100 * 1000
    print(f"  PyTorch argmax: {t_ref:.4f} ms "
          f"({'faster' if t_ref < t_ms else 'slower'}: "
          f"{max(t_ms, t_ref) / min(t_ms, t_ref):.2f}x)")


# PERFORMANCE NOTES
# =================
# - tl.max(x, axis=0, return_indices=True) → Triton 在 block 内用 warp shuffle
#   比较 (value, index) 对, 和 compare-and-swap 类似.
# - 跨 block 合并: 这里用 host 端合并 (候选数 = n/blocks ≈ 几十到几百),
#   对性能影响可忽略. 也可以用 atomic_max 做全 GPU 端合并.
# - Bandwidth: ~80-85% peak (memory-bound, 只读 4 bytes/elem).
# - 对比 13_vector_max: 多了 index tracking, 但性能几乎相同.
# - 应用: Beam search 的 top-k 选择, Attention 中的 argmax sampling.

if __name__ == "__main__":
    main()
