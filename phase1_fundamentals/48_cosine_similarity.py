"""
48_cosine_similarity.py — Cosine Similarity (x·y)/(|x||y|)

学习目标:
  - 掌握 cosine similarity 的 GPU 实现: norm + dot product 融合
  - 理解 "多统计量融合" 的 pattern: 一次 pass 算 sum(xy), sum(x²), sum(y²)
  - 学习数值稳定性处理: |x||y| ≈ 0 时的 div-by-zero 防护

数学定义:
  cos_sim(x, y) = sum(x[i] * y[i]) / (|x| * |y|)
  其中 |x| = sqrt(sum(x[i]²)), |y| = sqrt(sum(y[i]²))

为什么重要:
  - Text embeddings: 两个句子的语义相似度
  - Recommendation: user embedding · item embedding
  - Self-supervised learning (SimCLR): 对比学习的核心相似度度量
  - Attention clipping: 对 attention score 做 cosine normalization

和之前算子的关系:
  - 38_vector_dot: sum(x*y)  (dot product, 无归一化)
  - 14_vector_norm_l2: sqrt(sum(x²))  (L2 norm)
  - 48_cosine_similarity = dot / (norm_a * norm_b)  (组合)

实现策略:
  Pass 1 (GPU): 同时累加 xy_sum, x2_sum, y2_sum
  Pass 2 (host): cos = xy_sum / sqrt(x2_sum * y2_sum)

运行: python phase1_fundamentals/48_cosine_similarity.py
"""

import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch, triton, triton.language as tl


@triton.jit
def cosine_sim_kernel(x_ptr, y_ptr, xy_sum_ptr, x2_sum_ptr, y2_sum_ptr,
                       n_elements, BLOCK_SIZE: tl.constexpr):
    """
    一次 pass 累加三个统计量:
      xy_sum  = sum(x*y)
      x2_sum  = sum(x²)
      y2_sum  = sum(y²)
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # [GPU] 三个并行 reduction, 一次读取
    tl.atomic_add(xy_sum_ptr, tl.sum(x * y, axis=0))
    tl.atomic_add(x2_sum_ptr, tl.sum(x * x, axis=0))
    tl.atomic_add(y2_sum_ptr, tl.sum(y * y, axis=0))


def cosine_similarity(x: torch.Tensor, y: torch.Tensor,
                      eps: float = 1e-8) -> torch.Tensor:
    """
    Cosine similarity: sum(x*y) / (|x| * |y|).
    一次 GPU pass 累加所需统计量, host 端计算最终结果.
    """
    assert x.shape == y.shape
    n = x.numel()
    xy_sum = torch.zeros(1, device=x.device, dtype=torch.float32)
    x2_sum = torch.zeros(1, device=x.device, dtype=torch.float32)
    y2_sum = torch.zeros(1, device=x.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    cosine_sim_kernel[grid](x, y, xy_sum, x2_sum, y2_sum, n, BLOCK_SIZE=1024)
    # Host 端: cos = xy_sum / sqrt(x2_sum * y2_sum + eps)
    denom = torch.sqrt(x2_sum * y2_sum) + eps
    return xy_sum / denom


def main():
    print("=" * 60)
    print("48_cosine_similarity — Cosine Similarity")
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.manual_seed(42)

    for name, size in [("small ", 256), ("medium", 65536), ("large ", 1048576)]:
        a = torch.randn(size, device="cuda")
        b = torch.randn(size, device="cuda")
        cs_t = cosine_similarity(a, b).item()
        cs_r = torch.nn.functional.cosine_similarity(
            a.unsqueeze(0), b.unsqueeze(0)
        ).item()
        err = abs(cs_t - cs_r)
        print(f"  [{name}] size={size:8d}  cos_sim={cs_t:.6f}/{cs_r:.6f}  "
              f"err={err:.2e}  {'✅' if err < 1e-6 else '❌'}")

    # Demo: embedding similarity
    print("\n--- Embedding Similarity Demo ---")
    cat = torch.randn(128, device="cuda")
    dog = torch.randn(128, device="cuda")
    airplane = torch.randn(128, device="cuda")
    sim_cat_dog = cosine_similarity(cat, dog).item()
    sim_cat_air = cosine_similarity(cat, airplane).item()
    print(f"  cat · dog      : {sim_cat_dog:.4f}")
    print(f"  cat · airplane : {sim_cat_air:.4f}")
    print(f"  随机向量余弦相似度通常接近 0 (正交)")

    # Edge case: zero vector
    print("\n--- Edge case: zero vector (div-by-zero protection) ---")
    zero = torch.zeros(256, device="cuda")
    nonzero = torch.randn(256, device="cuda")
    cs_zero = cosine_similarity(zero, nonzero).item()
    print(f"  cos_sim(0, random) = {cs_zero:.6f} (should be ≈ 0, no NaN)")

    # Perf
    import time
    print("\n--- Performance ---")
    a = torch.randn(16777216, device="cuda", dtype=torch.float32)
    b = torch.randn(16777216, device="cuda", dtype=torch.float32)
    for _ in range(10): cosine_similarity(a, b)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100): cosine_similarity(a, b)
    torch.cuda.synchronize()
    t_ms = (time.perf_counter() - t0) / 100 * 1000
    bw = (a.numel() * 4 * 2) / (t_ms / 1000) / 1e9
    print(f"  Triton cosine_sim: {t_ms:.4f} ms, {bw:.1f} GB/s")
    t0 = time.perf_counter()
    for _ in range(100): torch.nn.functional.cosine_similarity(
        a.unsqueeze(0), b.unsqueeze(0))
    torch.cuda.synchronize()
    t_ref = (time.perf_counter() - t0) / 100 * 1000
    print(f"  PyTorch cosine_sim: {t_ref:.4f} ms")


# PERFORMANCE NOTES
# =================
# - Cosine similarity 需要 3 个统计量 (xy_sum, x²_sum, y²_sum),
#   但在一次 pass 中全部累加完成 (读 x, y 各一次).
# - atomic_add 3 次: 对 ~100 blocks 无显著冲突.
# - 和 15_welford / 29_parallel_mean_var 类似: GPU 累加, host 合并.
# - 可以扩展为 batch cosine_sim: N×M 行余弦相似度矩阵.

if __name__ == "__main__":
    main()
