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

import math
import torch
import triton
import triton.language as tl
from triton.testing import do_bench


# Activation type constants (integer, passed as tl.constexpr to kernel)
# 0=none, 1=relu, 2=gelu, 3=silu


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=3),
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


def unfused_matmul_bias_act(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor, activation: str = "none"
) -> torch.Tensor:
    """Separate kernels: matmul → bias → activation (3 HBM roundtrips)."""
    c = a @ b
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

        # Reference (unfused)
        c_ref = unfused_matmul_bias_act(a.float(), b.float(), bias.float(), activation=act).half()

        max_diff = (c_fused.float() - c_ref.float()).abs().max().item()
        tol = max(0.05, 0.005 * (K ** 0.5))
        status = "✅" if max_diff < tol else "❌"
        print(f"  max_diff = {max_diff:.6e} (tol={tol:.1e})  {status}")

    # Performance: fused vs unfused
    print(f"\n{'='*70}")
    print("Performance: Fused vs Unfused")
    print(f"{'='*70}")

    for M, N, K in [(1024, 4096, 4096), (2048, 4096, 4096)]:
        for act in ["gelu", "silu"]:
            a = torch.randn(M, K, device="cuda", dtype=torch.float16)
            b = torch.randn(K, N, device="cuda", dtype=torch.float16)
            bias = torch.randn(N, device="cuda", dtype=torch.float16)

            # Timing
            ms_fused = do_bench(lambda: fused_matmul(a, b, bias, activation=act))
            ms_unfused = do_bench(lambda: unfused_matmul_bias_act(a, b, bias, activation=act))

            speedup = ms_unfused / ms_fused
            label = f"{M}×{K}×{N} + {act}"
            print(f"  {label}: fused={ms_fused:.3f}ms  unfused={ms_unfused:.3f}ms  "
                  f"speedup={speedup:.2f}x")

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
# - Epilogue fusion 的核心价值: 减少 HBM 往返次数
# - 对于大矩阵 (M,N large), matmul 是 compute-bound, 融合收益不大
# - 对于中等矩阵 (M,N moderate), matmul 接近 ridge point, 融合减少
#   memory traffic 可以改善 10-30% 的延迟
# - [COMPILER] ACTIVATION=constexpr 使编译器在编译时消除所有 if-elif 分支:
#   - 每个 activation type 生成一份专用的 PTX (零运行时开销)
#   - 不同 activation 的 kernel 在 cache 中分别存储
# - GELU vs ReLU:
#   - ReLU: 只是 max(0, x) → 几乎没有额外开销
#   - GELU: tanh 近似需要 ~5 条指令 → 依然远小于 matmul
#   - SiLU: sigmoid + multiply → ~10 条指令
# - [GPU] bias load 被 BLOCK_M 个线程共享，L1 cache 命中率 ≈ 100%
# - 实际 GPU kernel 还常融合 dropout、residual add 等操作
# - 参考: NVIDIA cuBLASLt 的 epilogue、CUTLASS 的 epilogue visitor


if __name__ == "__main__":
    main()
