"""
02_matmul_tiled.py — 生产级分块 GEMM（shared memory + autotune）

学习目标：
  - 使用 shared memory 缓存 A/B tile（复用数据，减少 HBM 访问）
  - 理解 @triton.autotune 的多维度搜索空间
  - 掌握 num_warps, num_stages 的作用
  - 学会做 roofline 分析：compute bound vs memory bound

运行: python phase2_compute/02_matmul_tiled.py
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # 搜索空间: BLOCK_M × BLOCK_N × BLOCK_K × num_warps × num_stages
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_tiled_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Tiled MatMul with shared memory caching.

    核心思想:
      将 A 和 B 分块加载到 shared memory，然后从 shared memory 做 MMA。
      每次 K 维迭代:
        1. 加载 A[BLOCK_M, BLOCK_K] 到 shared memory
        2. 加载 B[BLOCK_K, BLOCK_N] 到 shared memory
        3. 从 shared memory 计算 acc += A_tile @ B_tile
      这样 A/B 的每个元素被复用 (C tile 的另一个维度 / BLOCK) 次。

    [COMPILER] num_stages > 1 时，Triton 会做 software pipelining:
      在计算当前 tile 的同时，异步预取下一个 tile 的 A/B。
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # C tile 的偏移 (在 M 和 N 维度)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # 累加器: [BLOCK_M, BLOCK_N]，保持在寄存器中
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # 沿 K 维迭代
    for k in range(0, K, BLOCK_K):
        # ---- Load A tile ----
        offs_k = k + tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        # [COMPILER] Triton 编译器会自动将 a 放入 shared memory
        # 如果 num_stages > 1，还会插入 cp.async 指令
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # ---- Load B tile ----
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # ---- MMA: acc += a @ b ----
        # tl.dot 在 Tensor Core 上映射为 mma.sync 指令
        acc += tl.dot(a, b)

    # Store C tile
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def matmul_tiled(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.dim() == 2 and b.dim() == 2
    M, K = a.shape
    K2, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    matmul_tiled_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


def main():
    print("=" * 60)
    print("02_matmul_tiled — Production GEMM with autotuning")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 测试多组大小
    sizes = [
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
    ]

    for M, N, K in sizes:
        a = torch.randn((M, K), device="cuda", dtype=torch.float16)
        b = torch.randn((K, N), device="cuda", dtype=torch.float16)

        # 正确性
        c_triton = matmul_tiled(a, b)
        c_torch = torch.mm(a, b)
        max_diff = (c_triton.float() - c_torch.float()).abs().max().item()

        # 性能
        n_iter = 100
        for _ in range(10):
            matmul_tiled(a, b)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_iter):
            matmul_tiled(a, b)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / n_iter
        tflops = (2 * M * N * K) / (ms * 1e-3) / 1e12

        status = "✅" if max_diff < 0.01 else "❌"
        print(f"  {M}x{N}x{K}: {ms:.4f}ms  {tflops:.1f} TFLOPS  diff={max_diff:.2e}  {status}")


# PERFORMANCE NOTES
# =================
# - 使用 shared memory 后，A/B 的每个元素被复用 BLOCK_N/BLOCK_M 次
# - 算术强度: O(BLOCK) — 更大的 BLOCK → 更高的算术强度
# - 但 BLOCK 受限于 shared memory 容量和寄存器
# - [COMPILER] num_stages:
#   - =1: 同步 load → compute → store（串行）
#   - =2: double buffering. load[i+1] 与 compute[i] 重叠
#   - =3: 三级流水，更深的延迟隐藏
# - [COMPILER] 软件流水线 = VLIW 的 modulo scheduling:
#   - 编译器展开循环 → 重排指令 → 插入 prefetch
# - Roofline 分析:
#   - H100 peak fp16: 989 TFLOPS, HBM: 3.35 TB/s
#   - Ridge point = 989e12 / 3.35e12 ≈ 295 FLOP/byte
#   - 对于大矩阵: 算术强度 ~ BLOCK/2 → 128/2 = 64 FLOP/byte < 295
#     → 理论上是 memory-bound
#   - 但实际上 shared memory 大幅减少 HBM 访问，实际性能接近 compute-bound
# - 用 ncu 分析: ncu --set full python phase2_compute/02_matmul_tiled.py
# - 后续优化方向:
#   - Warp specialization (Hopper): producer warp 做 load, consumer warp 做 MMA
#   - FP8 数据类型: 2x 吞吐量
#   - Persistent kernel: 减少 grid launch 开销


if __name__ == "__main__":
    main()
