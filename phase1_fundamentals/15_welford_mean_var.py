"""
15_welford_mean_var.py — Single-Pass Mean & Variance (Welford Algorithm)

学习目标:
  - 掌握 Welford 在线算法: 一次遍历同时计算 mean 和 variance
  - 理解为什么 Welford 比 2-pass (先 mean 再 var) 更高效
  - 为优化 LayerNorm (从 3-pass 到 1-pass) 打基础

Welford 算法 (在线更新):
  count = 0, mean = 0, M2 = 0
  for each x:
    count += 1
    delta = x - mean
    mean += delta / count
    delta2 = x - mean
    M2 += delta * delta2
  variance = M2 / count

为什么需要这个?
  - LayerNorm 当前是 3-pass (读 x 三次): mean → var → normalize
  - Welford 可以做到 1-pass: 一次遍历算出 mean + var
  - 减少 HBM 读 = 提升 bandwidth utilization

运行: python phase1_fundamentals/15_welford_mean_var.py
"""

import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch, triton, triton.language as tl
from utils.profiler import bench_compare, print_compare_report


@triton.jit
def welford_kernel(x_ptr, mean_ptr, var_ptr, n_elements,
                   BLOCK_SIZE: tl.constexpr):
    """
    Single-pass Welford: 一次遍历计算 mean 和 variance.
    每个 program 处理一段连续数据, 返回该段的 (count, mean, M2).
    Python wrapper 负责合并各段的 Welford 统计量.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Welford 在线更新 (在寄存器中)
    count = 0.0
    mean = 0.0
    M2 = 0.0
    for i in range(BLOCK_SIZE):
        val = tl.load(x_ptr + pid * BLOCK_SIZE + i)
        is_valid = (pid * BLOCK_SIZE + i) < n_elements
        if is_valid:
            count += 1.0
            delta = val - mean
            mean += delta / count
            delta2 = val - mean
            M2 += delta * delta2

    tl.store(mean_ptr + pid, mean)
    tl.store(var_ptr + pid, M2)


def welford_mean_var(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (mean, variance) — 单次遍历"""
    n = x.numel()
    n_blocks = triton.cdiv(n, 1024)
    means = torch.empty(n_blocks, device=x.device, dtype=torch.float32)
    M2s = torch.empty(n_blocks, device=x.device, dtype=torch.float32)
    grid = (n_blocks,)
    welford_kernel[grid](x, means, M2s, n, BLOCK_SIZE=1024)
    # 合并各 block 的 Welford 统计量 (Python 端, 标量计算)
    total_mean = means.mean()
    total_var = M2s.sum() / n
    return total_mean, total_var


def main():
    print("=" * 60)
    print("15_welford_mean_var — Single-Pass Mean & Variance")
    print("=" * 60)
    if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.manual_seed(42)
    for name, size in [("small", 256), ("medium", 4096), ("large", 65536)]:
        x = torch.randn(size, device="cuda")
        m_t, v_t = welford_mean_var(x)
        m_r, v_r = x.mean().item(), x.var(unbiased=False).item()
        print(f"  [{name}] size={size} mean={m_t:.4f}/{m_r:.4f} var={v_t:.4f}/{v_r:.4f} {'✅' if abs(m_t-m_r)<1e-3 and abs(v_t-v_r)<1e-3 else '❌'}")
    print("\n--- Performance vs 2-pass ---")
    x = torch.randn(16777216, device="cuda", dtype=torch.float32)
    def two_pass(): m=x.mean(); _=((x-m)**2).mean()
    n = x.numel()
    result = bench_compare({"Triton Welford 1-pass": lambda: welford_mean_var(x), "PyTorch 2-pass": two_pass}, flops=n*4, bytes_accessed=n*4, dtype="fp32")
    print_compare_report(result)

# PERFORMANCE NOTES
# =================
# - Welford 一次遍历计算 mean + var, 比 2-pass 减少 50% HBM 读
# - 但本实现的 sequential loop (BLOCK_SIZE 内逐个处理) 非常慢
# - 生产实现: 用 warp shuffle + shared memory 做 block 内并行 Welford
# - 应用: 优化 LayerNorm 从 3-pass → 1-pass (见 21_layer_norm.py 的 PERFORMANCE NOTES)

if __name__ == "__main__": main()
