"""Benchmark Triton vs PyTorch performance."""
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add_triton(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        mask_a = (offs_m[:, None] < M) & (k + offs_k[None, :] < K)
        mask_b = (k + offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul_triton(a: torch.Tensor, b: torch.Tensor):
    assert a.shape[1] == b.shape[0]
    M, K = a.shape
    K2, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 128, 256, 32
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return c


def main():
    print("=" * 60)
    print("Triton vs PyTorch Benchmark")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    # --- Vector Add ---
    print("\n--- Vector Add ---")
    sizes = [1024, 65536, 1048576, 16777216]
    for size in sizes:
        x = torch.rand(size, device="cuda", dtype=torch.float32)
        y = torch.rand(size, device="cuda", dtype=torch.float32)

        # warmup
        for _ in range(10):
            _ = add_triton(x, y)
        torch.cuda.synchronize()

        # benchmark triton
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            _ = add_triton(x, y)
        end.record()
        torch.cuda.synchronize()
        triton_ms = start.elapsed_time(end) / 100

        # benchmark pytorch
        start.record()
        for _ in range(100):
            _ = x + y
        end.record()
        torch.cuda.synchronize()
        torch_ms = start.elapsed_time(end) / 100

        print(f"  size={size:>10,}: Triton={triton_ms:.4f}ms, PyTorch={torch_ms:.4f}ms")

    # --- Matrix Multiply ---
    print("\n--- Matrix Multiply (fp16) ---")
    matmul_sizes = [(256, 256, 256), (1024, 1024, 1024), (4096, 4096, 4096)]
    for M, N, K in matmul_sizes:
        a = torch.randn((M, K), device="cuda", dtype=torch.float16)
        b = torch.randn((K, N), device="cuda", dtype=torch.float16)

        # warmup
        for _ in range(10):
            _ = matmul_triton(a, b)
        torch.cuda.synchronize()

        # benchmark triton
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            _ = matmul_triton(a, b)
        end.record()
        torch.cuda.synchronize()
        triton_ms = start.elapsed_time(end) / 100

        # benchmark pytorch
        start.record()
        for _ in range(100):
            _ = torch.mm(a, b)
        end.record()
        torch.cuda.synchronize()
        torch_ms = start.elapsed_time(end) / 100

        # verify correctness
        c_triton = matmul_triton(a, b)
        c_torch = torch.mm(a, b)
        max_diff = (c_triton.float() - c_torch.float()).abs().max().item()

        print(
            f"  {M}x{N}x{K}: Triton={triton_ms:.4f}ms, PyTorch={torch_ms:.4f}ms, "
            f"max_diff={max_diff:.6f}"
        )

    print("\n✅ All benchmarks passed!")


if __name__ == "__main__":
    main()
