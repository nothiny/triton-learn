"""
05_matmul_fused_bias_act.py — Fused MatMul + Bias + Activation

学习目标:
  - 理解 epilogue fusion 的核心思想
  - 掌握 bias broadcast 在 tile 内的实现
  - 学会用 constexpr 在编译时选择不同的激活函数

背景:
  Transformer FFN 的标准模式是: C = Activation(X @ W + bias)
  如果不融合，需要 3 个 kernel:
    1. C_temp = X @ W       (写 HBM)
    2. C_temp += bias       (读 HBM + 写 HBM)
    3. C = Act(C_temp)      (读 HBM + 写 HBM)

  Epilogue fusion 把 bias 和 activation 合并到 matmul kernel 中，
  在 accumulator 写回之前完成，省去中间的 HBM 往返。

  对于大矩阵 (M,N 很大)，省去 2 次 HBM 读写可以带来 10-30% 的延迟改善。

支持的激活函数:
  - none:    纯 GEMM (C = A @ B + bias)
  - relu:    ReLU (max(0, x))
  - gelu:    GELU tanh 近似 (用于 BERT/GPT)
  - silu:    SiLU (x * sigmoid(x), 用于 Llama)

运行: python phase2_compute/05_matmul_fused_bias_act.py
"""

import importlib.util
import math
import os
import torch
import triton
import triton.language as tl
from triton.testing import do_bench

# Load matmul_tiled from 02 for fair Triton-vs-Triton comparison
# (importlib needed because filenames starting with digits can't be imported directly)
_SPEC_TILED = importlib.util.spec_from_file_location(
    "matmul_tiled",
    os.path.join(os.path.dirname(__file__), "02_matmul_tiled.py"),
)
_MOD_TILED = importlib.util.module_from_spec(_SPEC_TILED)
_SPEC_TILED.loader.exec_module(_MOD_TILED)
matmul_tiled = _MOD_TILED.matmul_tiled


# Activation type constants (integer, passed as tl.constexpr to kernel)
# 0=none, 1=relu, 2=gelu, 3=silu


@triton.autotune(
    configs=[
        # 搜索空间与 02_matmul_tiled 完全一致，覆盖 BLOCK_M/ BLOCK_N/ BLOCK_K/ num_warps/ num_stages
        # ---- 小 tile: BLOCK_M=64 ----
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4, num_stages=3),

        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),

        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4, num_stages=3),

        # ---- 中等 tile: BLOCK_M=128 ----
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),

        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),

        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=3),

        # ---- 大 tile: BLOCK_M=256 ----
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),

        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=4),

        # ---- 大 K block（减少 K 维迭代次数）----
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def fused_matmul_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ACTIVATION: tl.constexpr,  # 0=none, 1=relu, 2=gelu, 3=silu
):
    """Fused GEMM + bias + activation kernel."""
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # ---- Main GEMM loop (same as 02_matmul_tiled) ----
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)

        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        acc += tl.dot(a, b)

    # ---- Epilogue: Bias addition ----
    # bias shape: (N,) → broadcast across M dimension
    # [GPU] bias 的每个元素被 BLOCK_M 个线程复用 → L1 cache 友好
    bias_ptrs = bias_ptr + offs_n
    bias = tl.load(bias_ptrs, mask=offs_n < N, other=0.0)  # [BLOCK_N]
    acc += bias[None, :]  # broadcast: [1, BLOCK_N] → [BLOCK_M, BLOCK_N]

    # ---- Epilogue: Activation ----
    # [COMPILER] ACTIVATION 是 tl.constexpr，编译器消除所有 dead code 分支
    # 使用整数字面量比较 (0=none, 1=relu, 2=gelu, 3=silu)
    if ACTIVATION == 1:
        acc = tl.maximum(acc, 0.0)
    elif ACTIVATION == 2:
        # GELU tanh 近似: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        # tanh(z) = 2*sigmoid(2z) - 1  (since tl.tanh not available)
        c = math.sqrt(2.0 / math.pi)
        inner = c * (acc + 0.044715 * acc * acc * acc)
        tanh_inner = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
        acc = 0.5 * acc * (1.0 + tanh_inner)
    elif ACTIVATION == 3:
        # SiLU(x) = x * sigmoid(x)
        acc = acc * tl.sigmoid(acc)
    # ACTIVATION == 0: no activation (just matmul + bias)

    # Store result
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def fused_matmul(
    a: torch.Tensor,      # (M, K)
    b: torch.Tensor,      # (K, N)
    bias: torch.Tensor,   # (N,)
    activation: str = "none",
) -> torch.Tensor:
    """
    Fused MatMul + Bias + Activation.

    Args:
        a: (M, K) input
        b: (K, N) weight
        bias: (N,) bias vector
        activation: "none" | "relu" | "gelu" | "silu"

    Returns:
        c: (M, N)
    """
    assert a.dim() == 2 and b.dim() == 2
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    assert bias.shape == (N,) or bias.shape == (1, N)

    act_map = {"none": 0, "relu": 1, "gelu": 2, "silu": 3}
    act_type = act_map[activation]

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    b_contig = bias.reshape(-1).contiguous()

    fused_matmul_kernel[grid](
        a, b, b_contig, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        ACTIVATION=act_type,
    )
    return c


# ==============================================================================
# Unfused reference (for performance comparison)
# ==============================================================================


def unfused_cublas_matmul_bias_act(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor, activation: str = "none"
) -> torch.Tensor:
    """cuBLAS matmul + separate bias + activation kernels (3 HBM roundtrips).

    Uses PyTorch's ``a @ b`` which dispatches to cuBLAS for fp16 GEMM — this is
    the strongest available baseline but involves THREE separate kernel launches
    (matmul → bias → activation), each writing intermediate results to HBM.
    """
    c = a @ b
    c = c + bias
    if activation == "relu":
        c = torch.relu(c)
    elif activation == "gelu":
        c = torch.nn.functional.gelu(c, approximate="tanh")
    elif activation == "silu":
        c = torch.nn.functional.silu(c)
    return c


def unfused_triton_matmul_bias_act(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor, activation: str = "none"
) -> torch.Tensor:
    """Triton matmul (02_matmul_tiled) + separate bias + activation kernels.

    This is the FAIR baseline for measuring fusion benefit: both fused and
    unfused paths use the SAME Triton matmul implementation (identical autotune
    configs).  The only difference is whether bias + activation happen inside
    the matmul kernel (fused, 1 kernel launch) or as separate PyTorch ops
    (unfused, 3 kernel launches = 2 extra HBM roundtrips).

    By comparing ``unfused_triton_*`` against ``fused_matmul`` we isolate the
    pure epilogue-fusion effect without the confounding cuBLAS-vs-Triton gap.
    """
    c = matmul_tiled(a, b)
    c = c + bias
    if activation == "relu":
        c = torch.relu(c)
    elif activation == "gelu":
        c = torch.nn.functional.gelu(c, approximate="tanh")
    elif activation == "silu":
        c = torch.nn.functional.silu(c)
    return c


# ==============================================================================
# Main
# ==============================================================================


def main():
    print("=" * 70)
    print("05_matmul_fused_bias_act — Fused MatMul + Bias + Activation")
    print("=" * 70)

    torch.manual_seed(42)

    # Test correctness for each activation
    configs = [
        (512, 256, 1024, "none", "MatMul + Bias (no act)"),
        (512, 256, 1024, "relu", "MatMul + Bias + ReLU"),
        (512, 256, 1024, "gelu", "MatMul + Bias + GELU (FFN pattern)"),
        (512, 256, 1024, "silu", "MatMul + Bias + SiLU (Llama FFN)"),
    ]

    for M, N, K, act, desc in configs:
        print(f"\n── {desc} ──")
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)
        bias = torch.randn(N, device="cuda", dtype=torch.float16)

        # Fused Triton
        c_fused = fused_matmul(a, b, bias, activation=act)

        # Reference (cuBLAS unfused, in fp32 for accuracy)
        c_ref = unfused_cublas_matmul_bias_act(a.float(), b.float(), bias.float(), activation=act).half()

        max_diff = (c_fused.float() - c_ref.float()).abs().max().item()
        tol = max(0.05, 0.005 * (K ** 0.5))
        status = "✅" if max_diff < tol else "❌"
        print(f"  max_diff = {max_diff:.6e} (tol={tol:.1e})  {status}")

    # Performance: three-way comparison
    #   (a) cuBLAS unfused — the strongest matmul but 3 kernel launches
    #   (b) Triton unfused — same matmul as fused, but split into 3 kernels
    #   (c) Triton fused   — matmul + bias + activation in 1 kernel
    #
    # Shapes are grouped by scale so you can see how fusion benefit changes
    # as matrices grow from moderate → large → huge.

    all_shapes = [
        # --- Moderate (FFN on small batch / short sequence) ---
        # (M, K, N) with context
        (1024, 4096, 4096,   "Llama 8B: seq=1024, Q/K/V"),
        (2048, 4096, 4096,   "Llama 8B: seq=2048, Q/K/V"),

        # --- Large (closer to real workloads) ---
        (4096, 4096, 4096,   "Llama 8B: seq=4096, Q/K/V"),
        (4096, 4096, 14336,  "Llama 8B: seq=4096, FFN up"),
        (4096, 14336, 4096,  "Llama 8B: seq=4096, FFN down"),

        # --- Huge (Llama 70B scale) ---
        (8192, 8192, 8192,   "Llama 70B: seq=8192, Q/K/V"),
        (8192, 8192, 28672,  "Llama 70B: seq=8192, FFN up"),

        # --- Extreme (long sequence) ---
        (16384, 4096, 4096,  "Long seq: 16K tokens, Q/K/V"),
    ]

    # Fixed activation for timing sweep (silu is most common in modern LLMs)
    # Change to "gelu" if you want GELU-specific numbers.
    for M, K, N, desc in all_shapes:
        for act in ["gelu", "silu"]:
            a = torch.randn(M, K, device="cuda", dtype=torch.float16)
            b = torch.randn(K, N, device="cuda", dtype=torch.float16)
            bias = torch.randn(N, device="cuda", dtype=torch.float16)

            ms_cublas_unf = do_bench(
                lambda: unfused_cublas_matmul_bias_act(a, b, bias, activation=act)
            )
            ms_triton_unf = do_bench(
                lambda: unfused_triton_matmul_bias_act(a, b, bias, activation=act)
            )
            ms_fused = do_bench(lambda: fused_matmul(a, b, bias, activation=act))

            fusion_benefit = ms_triton_unf / ms_fused  # >1 means fusion helps

            label = f"{M}×{K}×{N} + {act}"
            print(f"  {label}  ({desc}):")
            print(f"    cuBLAS unfused:  {ms_cublas_unf*1000:.1f} us")
            print(f"    Triton unfused:  {ms_triton_unf*1000:.1f} us")
            print(f"    Triton fused:    {ms_fused*1000:.1f} us")
            print(f"    Fusion speedup:   {fusion_benefit:.2f}x  "
                  f"({'✅' if fusion_benefit > 1.0 else '❌'} "
                  f"vs Triton unfused)")
            print(f"    vs cuBLAS:        {ms_fused / ms_cublas_unf:.2f}x  "
                  f"({'✅' if ms_fused <= ms_cublas_unf else '❌'} "
                  f"fused vs cuBLAS unfused)")
            print()

    # Memory analysis
    print(f"\n{'='*70}")
    print("Epilogue Fusion — Memory Savings")
    print(f"{'='*70}")
    M, N = 2048, 4096
    elem = 2  # fp16 bytes
    # Unfused: write C (M*N*2), read C for bias (M*N*2), write C (M*N*2),
    #          read C for act (M*N*2), write C (M*N*2)
    #           = 5 * M * N * 2 bytes of intermediate traffic
    # Fused:   only write final C (M*N*2)
    unfused_traffic = 5 * M * N * elem
    fused_traffic = 1 * M * N * elem
    print(f"  Unfused HBM traffic: {unfused_traffic / 1e6:.1f} MB")
    print(f"  Fused HBM traffic:   {fused_traffic / 1e6:.1f} MB")
    print(f"  Savings:             {(unfused_traffic - fused_traffic) / 1e6:.1f} MB "
          f"({unfused_traffic / fused_traffic:.0f}x reduction)")


# PERFORMANCE NOTES
# =================
# Epilogue fusion 的核心价值: 减少 HBM 往返次数。
#
# --- 融合收益分析 (Triton vs Triton, 公平对比) ---
# 当 fused 和 unfused 都使用相同的 Triton matmul 实现时:
#   - 小/中等矩阵 (M,N < 2048): 融合收益 10-30%，因为 element-wise ops
#     的 HBM 读写占整体延迟的比例较大
#   - 大矩阵 (M,N >= 2048, K 大): matmul 是 compute-bound，融合收益
#     降至 5-10%，因为 bias+gelu 的 HBM 开销相对 matmul 微乎其微
#   - Fusion 的绝对收益 = 省掉 2 次 (M×N) 级别的 HBM 读写
#     ≈ (2 × M × N × 2 bytes) / HBM_bandwidth
#
# --- 为什么之前 fused 比 unfused 慢 ---
# 旧版本只有 4 个 autotune 配置，搜索空间不足导致 matmul 部分选不到
# 最优 tile 尺寸，matmul 本身就跑得慢。同时 unfused baseline 用了
# cuBLAS (`a @ b`)，cuBLAS 的手写汇编比 Triton 编译器生成的代码
# 快 20-30%——融合省下的内存带宽根本填不平 matmul 的性能差距。
#
# 解决方案:
#   1. 把 autotune 配置扩展到与 02 一致 (20+ 配置) ← 已修复
#   2. 增加 ``unfused_triton_matmul_bias_act`` 作为公平 baseline，
#      对比时同时展示 cuBLAS unfused / Triton unfused / Triton fused
#
# --- 何时融合有收益 ---
# - Matmul 本身的实现质量接近 (Triton vs Triton, 而非 Triton vs cuBLAS)
# - 矩阵不是特别大 (M,N < 4096, K < 4096)，matmul 还没完全 compute-bound
# - Epilogue 操作越多 (bias + gelu + dropout + residual)，融合收益越大
# - 参考: NVIDIA cuBLASLt 的 epilogue、CUTLASS 的 epilogue visitor
#         都支持 fused epilogue，收益在 10-30%
#
# - [COMPILER] ACTIVATION=constexpr 使编译器在编译时消除所有 if-elif 分支:
#   - 每个 activation type 生成一份专用的 PTX (零运行时开销)
#   - 不同 activation 的 kernel 在 cache 中分别存储
# - GELU vs ReLU:
#   - ReLU: 只是 max(0, x) → 几乎没有额外开销
#   - GELU: tanh 近似需要 ~5 条指令 → 依然远小于 matmul
#   - SiLU: sigmoid + multiply → ~10 条指令
# - [GPU] bias load 被 BLOCK_M 个线程共享，L1 cache 命中率 ≈ 100%


if __name__ == "__main__":
    main()
