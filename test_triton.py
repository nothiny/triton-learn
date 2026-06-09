"""Test Triton kernel execution on GPU."""
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


def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


def main():
    print(f"Triton version: {triton.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device="cuda")
    y = torch.rand(size, device="cuda")

    output_triton = add(x, y)
    output_torch = x + y

    max_diff = torch.max(torch.abs(output_triton - output_torch)).item()
    all_close = torch.allclose(output_triton, output_torch)

    print(f"Input size: {size}")
    print(f"Triton output (first 5): {output_triton[:5].tolist()}")
    print(f"Torch output (first 5):  {output_torch[:5].tolist()}")
    print(f"Max difference: {max_diff}")
    print(f"All close: {all_close}")

    if all_close:
        print("\n✅ Triton kernel executed successfully!")
    else:
        print("\n❌ Triton kernel result mismatch!")


if __name__ == "__main__":
    main()
