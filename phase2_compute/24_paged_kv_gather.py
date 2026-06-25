"""
22_paged_kv_gather.py — PagedAttention: Gather KV from Paged Memory

学习目标:
  - 理解 vLLM/paged attention 的内存管理策略
  - 掌握通过 page table 做 indirect memory access 的模式
  - 学习 scatter/gather 在 GPU 上的高效实现

PagedAttention 背景:
  在 LLM 推理中，KV cache 占用主要显存。PagedAttention 将 KV cache
  分割为固定大小的 "pages"，通过 page table 映射逻辑位置到物理 page。

  简化场景:
    输入: pages (num_pages, page_size, n_heads, head_dim)
    查询: page_table (batch, max_pages) — 逻辑位置 → page index
    输出: kv_buffer (batch, seq_len, n_heads, head_dim)

运行: python phase2_compute/22_paged_kv_gather.py
"""

import torch
import triton
import triton.language as tl
from triton.testing import do_bench


@triton.jit
def paged_kv_gather_kernel(
    pages_ptr,          # (num_pages, page_size, n_heads, head_dim)
    page_table_ptr,     # (batch, max_pages) — int32, -1 = invalid
    output_ptr,         # (batch, seq_len, n_heads, head_dim)
    page_size,
    n_heads,
    head_dim,
    seq_len,            # = max_pages * page_size
    max_pages,
    stride_pg_p,        # page stride (dim 0)
    stride_pg_t,        # token-in-page stride (dim 1)
    stride_pg_h,        # head stride (dim 2)
    stride_out_b,       # batch stride (dim 0)
    stride_out_s,       # seq stride (dim 1)
    stride_out_h,       # head stride (dim 2)
    BLOCK_D: tl.constexpr,
):
    """
    Gather KV from paged memory to contiguous buffer.

    Grid: (batch * seq_len, n_heads)

    每个 program 处理一个 (batch, logical_token, head_id) 三元组，
    拷贝 head_dim 个元素。
    """
    pid_seq = tl.program_id(axis=0)
    head_id = tl.program_id(axis=1)

    batch_id = pid_seq // seq_len
    logical_pos = pid_seq % seq_len

    # 查询 page table
    page_id = logical_pos // page_size

    # 边界检查: page_id 不能超出 max_pages
    if page_id < max_pages:
        page_idx = tl.load(page_table_ptr + batch_id * max_pages + page_id)

        # 只处理有效 page
        if page_idx >= 0:
            offset_in_page = logical_pos % page_size

            offs_d = tl.arange(0, BLOCK_D)
            d_mask = offs_d < head_dim

            # 物理地址: pages[page_idx, offset_in_page, head_id, :]
            phys_offset = (page_idx * stride_pg_p +
                           offset_in_page * stride_pg_t +
                           head_id * stride_pg_h)
            in_ptrs = pages_ptr + phys_offset + offs_d
            data = tl.load(in_ptrs, mask=d_mask, other=0.0)

            # 逻辑地址: output[batch_id, logical_pos, head_id, :]
            out_offset = (batch_id * stride_out_b +
                          logical_pos * stride_out_s +
                          head_id * stride_out_h)
            out_ptrs = output_ptr + out_offset + offs_d
            tl.store(out_ptrs, data, mask=d_mask)


def paged_kv_gather(
    pages: torch.Tensor,
    page_table: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """
    pages: (num_pages, page_size, n_heads, head_dim)
    page_table: (batch, max_pages) — int32, -1 = invalid page
    """
    num_pages, page_size, n_heads, head_dim = pages.shape
    batch, max_pages = page_table.shape
    seq_len = max_pages * page_size

    output = torch.empty(batch, seq_len, n_heads, head_dim,
                         device=pages.device, dtype=pages.dtype)

    BLOCK_D = min(128, triton.next_power_of_2(head_dim))
    grid = (batch * seq_len, n_heads)

    paged_kv_gather_kernel[grid](
        pages, page_table, output,
        page_size, n_heads, head_dim, seq_len, max_pages,
        pages.stride(0), pages.stride(1), pages.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        BLOCK_D=BLOCK_D,
    )
    return output


def ref_paged_kv_gather(
    pages: torch.Tensor,
    page_table: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """PyTorch reference (CPU-safe, no .item() on GPU tensor)"""
    _, _, n_heads, head_dim = pages.shape
    batch, max_pages = page_table.shape
    seq_len = max_pages * page_size

    # Copy to CPU for reference
    pt_cpu = page_table.cpu()
    pages_cpu = pages.cpu()

    output = torch.zeros(batch, seq_len, n_heads, head_dim,
                         device="cpu", dtype=pages_cpu.dtype)

    for b in range(batch):
        for p in range(max_pages):
            page_idx = pt_cpu[b, p].item()
            if page_idx >= 0:
                for t in range(page_size):
                    logical_pos = p * page_size + t
                    output[b, logical_pos] = pages_cpu[page_idx, t]

    return output.to(pages.device)


def main():
    print("=" * 60)
    print("22_paged_kv_gather — Paged KV Cache Gather")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    configs = [
        (16, 2, 4, 32, 64),     # 16 pages, batch=2, page_size=4, n_heads=32, d=64
        (32, 2, 4, 64, 128),    # 32 pages, batch=2, page_size=4, n_heads=64, d=128
    ]

    for num_pages, batch, page_size, n_heads, head_dim in configs:
        pages = torch.randn(num_pages, page_size, n_heads, head_dim,
                            device="cuda", dtype=torch.float16)

        max_pages = 8
        page_table = torch.randint(0, num_pages, (batch, max_pages),
                                   device="cuda", dtype=torch.int32)
        # 随机 mask 掉一些 page
        mask = torch.rand(batch, max_pages, device="cuda") > 0.2
        page_table = torch.where(mask, page_table,
                                 torch.tensor(-1, device="cuda", dtype=torch.int32))

        out_triton = paged_kv_gather(pages, page_table, page_size)
        out_ref = ref_paged_kv_gather(pages.clone(), page_table.clone(), page_size)
        max_diff = (out_triton.float() - out_ref.float()).abs().max().item()

        ms = do_bench(lambda: paged_kv_gather(pages, page_table, page_size))
        seq_len = max_pages * page_size
        total_elems = batch * seq_len * n_heads * head_dim
        bandwidth = (total_elems * 2 * 2) / (ms * 1e-3) / 1e9  # read+write, fp16

        status = "✅" if max_diff < 0.01 else "❌"
        print(f"  {num_pages}p×{page_size}s×{n_heads}h×{head_dim}d: "
              f"{ms:.4f}ms  {bandwidth:.1f} GB/s  diff={max_diff:.2e}  {status}")

    print(f"\n  💡 Paged KV 的核心优势:")
    print(f"     - 物理 pages 可以不连续 → 减少碎片")
    print(f"     - 不同请求共享同一 page → KV cache 复用")
    print(f"     - Page table 间接寻址允许动态内存分配")


# PERFORMANCE NOTES
# =================
# - 纯 memory-bound 操作（无计算）
# - 性能瓶颈: page table 随机访问 + 非 coalesced page 读取
# - vLLM 生产实现将 gather 与 attention 融合（PagedAttention kernel）


if __name__ == "__main__":
    main()
