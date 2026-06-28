"""
phase3_production/09_chunk_cumprod_householder.py — Chunked Cumprod Householder

Forward and backward kernels for the cumulative product of Householder reflections,
a core building block of the Householder variant of the Gated Delta Rule.

Algorithm (forward, per split of size S):
  For each small chunk (size BT) within the split, processed in REVERSE order:
    1. Store current h → hc_suffix (suffix cumulative product)
    2. k_new = k - k @ h^T          (subtract future-component projection)
    3. v_new = w1 - h @ w1          (apply reflector to w1)
    4. h += v_new @ w2              (accumulate into cumulative reflector)
  Store final h → hc_whole (full split cumulative product)

The reflector at each step is I - w1 @ w2^T. The cumulative product
h = Π_{i=t...end} (I - w1_i @ w2_i^T) orthogonalizes the key projections.

Reference: fla-org/flash-linear-attention
  chunk_cumprod_householder_fwd_kernel / _bwd_kernel

运行: python phase3_production/09_chunk_cumprod_householder.py
"""

import math

import torch
import triton
import triton.language as tl
from triton.testing import do_bench


# ============================================================================
# Utilities
# ============================================================================


def _prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Build (n_chunks, 2) index table for varlen sequences."""
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    n_chunks_total = (lengths + chunk_size - 1) // chunk_size
    total_chunks = n_chunks_total.sum().item()
    out = torch.empty(total_chunks, 2, dtype=torch.int64, device=cu_seqlens.device)
    offset = 0
    for i in range(len(lengths)):
        nc = n_chunks_total[i].item()
        out[offset : offset + nc, 0] = i
        out[offset : offset + nc, 1] = torch.arange(nc, dtype=torch.int64)
        offset += nc
    return out


def _prepare_chunk_offsets(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Cumulative chunk offsets for varlen sequences."""
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    n_chunks = (lengths + chunk_size - 1) // chunk_size
    offsets = torch.zeros(len(lengths) + 1, dtype=torch.int64, device=cu_seqlens.device)
    offsets[1:] = n_chunks.cumsum(dim=0)
    return offsets


# ============================================================================
# Forward Kernel
# ============================================================================


@triton.jit
def _chunk_cumprod_householder_fwd_kernel(
    # Pointers
    k_ptr,          # (B, T, H, K) — input key
    k_new_ptr,      # (B, T, H, K) — output: k with future components removed
    w1_ptr,         # (B, T, H, K) — first Householder factor
    w2_ptr,         # (B, T, H, K) — second Householder factor
    hc_suffix_ptr,  # (NT, H, K, K) — intermediate h at small chunk boundaries
    hc_whole_ptr,   # (NS, H, K, K) — final h at split boundaries
    cu_seqlens_ptr,     # (N+1,) | None
    split_indices_ptr,  # (NS, 2) — maps split idx → (seq_idx, split_idx)
    chunk_offsets_ptr,  # (N+1,) — per-seq small chunk offsets
    split_offsets_ptr,  # (N+1,) — per-seq split offsets
    # Shapes
    BT: tl.constexpr,
    K: tl.constexpr,
    H: tl.constexpr,
    BK: tl.constexpr,
    T: tl.constexpr,
    S: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Forward pass — cumulative product of Householder reflectors.

    Grid: (NS, H)
      axis=0: split index within batch
      axis=1: head index

    Processes each split in reverse small-chunk order, accumulating h.
    """
    i_ss, i_h = tl.program_id(0), tl.program_id(1)

    # ── Sequence boundaries ──────────────────────────────────────
    if IS_VARLEN:
        i_n = tl.load(split_indices_ptr + i_ss * 2).to(tl.int32)
        i_s = tl.load(split_indices_ptr + i_ss * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens_ptr + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens_ptr + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
        boh = tl.load(chunk_offsets_ptr + i_n).to(tl.int32)
        boh_large = tl.load(split_offsets_ptr + i_n).to(tl.int32)
    else:
        NS_total = tl.cdiv(T, S)
        i_n = i_ss // NS_total
        i_s = i_ss % NS_total
        bos = (i_n * T).to(tl.int64)
        boh = i_n * tl.cdiv(T, BT)
        boh_large = i_n * tl.cdiv(T, S)

    NT_small = tl.cdiv(tl.minimum(S, T - i_s * S), BT)
    stride_h = H * K * K

    # ── Pointer offsets ──────────────────────────────────────────
    hc_whole_ptr += ((boh_large + i_s) * H + i_h) * K * K
    hc_suffix_ptr += ((boh + tl.cdiv(i_s * S, BT)) * H + i_h) * K * K
    k_ptr += (bos * H + i_h) * K
    k_new_ptr += (bos * H + i_h) * K
    w1_ptr += (bos * H + i_h) * K
    w2_ptr += (bos * H + i_h) * K

    # ── Register-resident hidden state: [K, K] ───────────────────
    b_h = tl.zeros([BK, BK], dtype=tl.float32)

    # Process small chunks in REVERSE order
    for i_t_small in range(NT_small - 1, -1, -1):
        # Store current h as the suffix product
        p_hc_suffix = tl.make_block_ptr(
            hc_suffix_ptr + i_t_small * stride_h, (K, K), (K, 1),
            (0, 0), (BK, BK), (1, 0),
        )
        tl.store(p_hc_suffix, b_h.to(hc_suffix_ptr.dtype.element_ty), boundary_check=(0, 1))

        # Load k for this small chunk: [BT, BK]
        p_k = tl.make_block_ptr(
            k_ptr, (T, K), (H * K, 1),
            (i_s * S + i_t_small * BT, 0), (BT, BK), (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # k_new = k - k @ h^T  (remove future-component projection)
        b_k_new = b_k - tl.dot(b_k, tl.trans(b_h.to(b_k.dtype)))

        # Load w1 (transposed view): [BK, BT], load w2: [BT, BK]
        p_w1 = tl.make_block_ptr(
            w1_ptr, (K, T), (1, H * K),
            (0, i_s * S + i_t_small * BT), (BK, BT), (0, 1),
        )
        p_w2 = tl.make_block_ptr(
            w2_ptr, (T, K), (H * K, 1),
            (i_s * S + i_t_small * BT, 0), (BT, BK), (1, 0),
        )
        b_w1 = tl.load(p_w1, boundary_check=(0, 1))
        b_w2 = tl.load(p_w2, boundary_check=(0, 1))

        # v_new = w1 - h @ w1: [BK, BK] @ [BK, BT] → [BK, BT]
        # Store in w2 dtype, then use w2 for the outer product
        b_v_new = (b_w1 - tl.dot(b_h.to(b_w1.dtype), b_w1)).to(b_w2.dtype)
        # h += v_new @ w2: [BK, BT] @ [BT, BK] → [BK, BK]
        b_h += tl.dot(b_v_new, b_w2)

        # Store k_new
        p_k_new = tl.make_block_ptr(
            k_new_ptr, (T, K), (H * K, 1),
            (i_s * S + i_t_small * BT, 0), (BT, BK), (1, 0),
        )
        tl.store(p_k_new, b_k_new.to(k_new_ptr.dtype.element_ty), boundary_check=(0, 1))

    # Store final h for this split
    p_hc_whole = tl.make_block_ptr(
        hc_whole_ptr, (K, K), (K, 1),
        (0, 0), (BK, BK), (1, 0),
    )
    tl.store(p_hc_whole, b_h.to(hc_whole_ptr.dtype.element_ty), boundary_check=(0, 1))


# ============================================================================
# Backward Kernel
# ============================================================================


@triton.jit
def _chunk_cumprod_householder_bwd_kernel(
    hc_suffix_ptr,     # (NT, H, K, K)
    dhc_whole_ptr,     # (NS, HQ, K, K) — gradient of final h
    k_ptr,             # (B, T, H, K)
    dk_ptr,            # (B, T, HQ, K) — gradient w.r.t. k_new
    w1_ptr,            # (B, T, H, K)
    w2_ptr,            # (B, T, H, K)
    dw1_ptr,           # (B, T, HQ, K) — output: gradient w.r.t. w1
    dw2_ptr,           # (B, T, HQ, K) — output: gradient w.r.t. w2
    dk_new_ptr,        # (B, T, HQ, K) — output: dk propagated through reflector
    cu_seqlens_ptr,
    split_indices_ptr,
    chunk_offsets_ptr,
    split_offsets_ptr,
    BT: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    T: tl.constexpr,
    S: tl.constexpr,
    G: tl.constexpr,       # GQA ratio: HQ // H
    H: tl.constexpr,
    HQ: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Backward pass — reverse-mode differentiation of the cumprod householder.

    Grid: (NS, HQ)
      axis=0: split index
      axis=1: query head (HQ = G * H for GQA)

    Processes small chunks in FORWARD order (reverse of forward).
    """
    i_ss, i_hq = tl.program_id(0), tl.program_id(1)
    i_h = i_hq // G  # GQA: map query head to key head

    # ── Sequence boundaries ──────────────────────────────────────
    if IS_VARLEN:
        i_n = tl.load(split_indices_ptr + i_ss * 2).to(tl.int32)
        i_s = tl.load(split_indices_ptr + i_ss * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens_ptr + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens_ptr + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
        boh = tl.load(chunk_offsets_ptr + i_n).to(tl.int32)
        boh_large = tl.load(split_offsets_ptr + i_n).to(tl.int32)
    else:
        NS_total = tl.cdiv(T, S)
        i_n = i_ss // NS_total
        i_s = i_ss % NS_total
        bos = (i_n * T).to(tl.int64)
        boh = i_n * tl.cdiv(T, BT)
        boh_large = i_n * tl.cdiv(T, S)

    NT_small = tl.cdiv(tl.minimum(S, T - i_s * S), BT)
    stride_h = H * K * K

    # ── Pointer offsets ──────────────────────────────────────────
    dhc_whole_ptr += ((boh_large + i_s) * HQ + i_hq) * K * K
    hc_suffix_ptr += ((boh + tl.cdiv(i_s * S, BT)) * H + i_h) * K * K
    k_ptr += (bos * H + i_h) * K
    w1_ptr += (bos * H + i_h) * K
    w2_ptr += (bos * H + i_h) * K
    dw1_ptr += (bos * HQ + i_hq) * K
    dw2_ptr += (bos * HQ + i_hq) * K
    dk_ptr += (bos * HQ + i_hq) * K
    dk_new_ptr += (bos * HQ + i_hq) * K

    # Load gradient of final h: [BK, BK]
    p_dhc_whole = tl.make_block_ptr(
        dhc_whole_ptr, (K, K), (K, 1), (0, 0), (BK, BK), (1, 0),
    )
    b_dhc = tl.load(p_dhc_whole, boundary_check=(0, 1)).to(tl.float32)

    # Process small chunks in FORWARD order
    for i_t_small in range(0, NT_small):
        t_offset = i_s * S + i_t_small * BT

        # Load saved forward intermediates
        p_k = tl.make_block_ptr(
            k_ptr, (T, K), (H * K, 1), (t_offset, 0), (BT, BK), (1, 0),
        )
        p_hc = tl.make_block_ptr(
            hc_suffix_ptr + i_t_small * stride_h, (K, K), (K, 1),
            (0, 0), (BK, BK), (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
        b_hc = tl.load(p_hc, boundary_check=(0, 1)).to(tl.float32)

        # Load upstream gradient dk: [BT, BK]
        p_dk = tl.make_block_ptr(
            dk_ptr, (T, K), (HQ * K, 1), (t_offset, 0), (BT, BK), (1, 0),
        )
        b_dk = tl.load(p_dk, boundary_check=(0, 1)).to(tl.float32)

        # dk_new = dk - dk @ hc   (propagate gradient through k ← k_new)
        b_dk_new = b_dk - tl.dot(b_dk.to(b_hc.dtype), b_hc)
        p_dk_new = tl.make_block_ptr(
            dk_new_ptr, (T, K), (HQ * K, 1), (t_offset, 0), (BT, BK), (1, 0),
        )
        tl.store(p_dk_new, b_dk_new.to(dk_new_ptr.dtype.element_ty), boundary_check=(0, 1))

        # Load w1, w2
        p_w1 = tl.make_block_ptr(
            w1_ptr, (T, K), (H * K, 1), (t_offset, 0), (BT, BK), (1, 0),
        )
        p_w2 = tl.make_block_ptr(
            w2_ptr, (T, K), (H * K, 1), (t_offset, 0), (BT, BK), (1, 0),
        )
        b_w1 = tl.load(p_w1, boundary_check=(0, 1)).to(tl.float32)
        b_w2 = tl.load(p_w2, boundary_check=(0, 1)).to(tl.float32)

        # dh = dhc - hc^T @ dhc  (gradient w.r.t. the suffix product)
        b_dh = b_dhc - tl.dot(tl.trans(b_hc), b_dhc.to(b_hc.dtype))

        # dw2 = w1 @ dh   and   dw1 = w2 @ dh^T
        b_dw2 = tl.dot(b_w1, b_dh.to(b_w1.dtype))
        b_dw1 = tl.dot(b_w2, tl.trans(b_dh.to(b_w2.dtype)))

        p_dw1 = tl.make_block_ptr(
            dw1_ptr, (T, K), (HQ * K, 1), (t_offset, 0), (BT, BK), (1, 0),
        )
        p_dw2 = tl.make_block_ptr(
            dw2_ptr, (T, K), (HQ * K, 1), (t_offset, 0), (BT, BK), (1, 0),
        )
        tl.store(p_dw1, b_dw1.to(dw1_ptr.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dw2, b_dw2.to(dw2_ptr.dtype.element_ty), boundary_check=(0, 1))

        # Update dhc for NEXT iteration (moving backwards through time):
        # dhc = dhc - (dhc @ w2^T) @ w1 - dk^T @ k
        b_dhc = b_dhc - tl.dot(
            tl.dot(b_dhc.to(b_w2.dtype), tl.trans(b_w2)).to(b_w1.dtype), b_w1,
        )
        b_dhc -= tl.dot(tl.trans(b_dk).to(b_k.dtype), b_k)


# ============================================================================
# Python Wrappers
# ============================================================================


def chunk_cumprod_householder_fwd(
    k: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    S: int,
    BT: int,
    cu_seqlens: torch.Tensor | None = None,
    split_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward pass: cumulative product of Householder reflectors.

    Args:
        k:  (B, T, H, K) key tensor
        w1: (B, T, H, K) first Householder factor
        w2: (B, T, H, K) second Householder factor
        S:  split size (large chunk)
        BT: small chunk size
        cu_seqlens: (N+1,) | None — for varlen
        split_indices: (NS, 2) | None — precomputed split mapping

    Returns:
        k_new:      (B, T, H, K) — k with future components removed
        hc_suffix:  (NT, H, K, K) — suffix cumulative products
        hc_whole:   (NS, H, K, K) — full split cumulative products
    """
    B, T, H, K = k.shape
    dtype = w1.dtype

    if split_indices is None and cu_seqlens is not None:
        split_indices = _prepare_chunk_indices(cu_seqlens, S)

    if cu_seqlens is None:
        N = B
        NS = N * triton.cdiv(T, S)
        NT = N * triton.cdiv(T, BT)
        chunk_offsets_ptr = None
        split_offsets_ptr = None
    else:
        N = len(cu_seqlens) - 1
        chunk_offsets = _prepare_chunk_offsets(cu_seqlens, BT)
        split_offsets = _prepare_chunk_offsets(cu_seqlens, S)
        chunk_offsets_ptr = chunk_offsets
        split_offsets_ptr = split_offsets
        NS = int(split_offsets[-1].item())
        NT = int(chunk_offsets[-1].item())

    grid = (NS, H)
    hc_whole = torch.empty(NS, H, K, K, device=k.device, dtype=dtype)
    k_new = torch.empty_like(k, dtype=k.dtype)
    hc_suffix = torch.empty(NT, H, K, K, device=k.device, dtype=dtype)

    # K=128 needs more warps to avoid correctness issues (empirical finding from FLA)
    num_warps = 8 if K == 128 else 4

    _chunk_cumprod_householder_fwd_kernel[grid](
        k, k_new, w1, w2, hc_suffix, hc_whole,
        cu_seqlens, split_indices,
        chunk_offsets_ptr, split_offsets_ptr,
        BT=BT, K=K, H=H, BK=K, T=T, S=S,
        IS_VARLEN=(cu_seqlens is not None),
        num_warps=num_warps,
        num_stages=3,
    )
    return k_new, hc_suffix, hc_whole


def chunk_cumprod_householder_bwd(
    w1: torch.Tensor,
    w2: torch.Tensor,
    hc_suffix: torch.Tensor,
    dhc_whole: torch.Tensor,
    k: torch.Tensor,
    dk: torch.Tensor,
    S: int,
    BT: int,
    cu_seqlens: torch.Tensor | None = None,
    split_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward pass: gradients of the cumprod householder.

    Args:
        w1:         (B, T, H, K)
        w2:         (B, T, H, K)
        hc_suffix:  (NT, H, K, K) — saved forward intermediates
        dhc_whole:  (NS, HQ, K, K) — gradient of final h
        k:          (B, T, H, K) — original key
        dk:         (B, T, HQ, K) — upstream gradient w.r.t. k_new
        S:          split size
        BT:         small chunk size
        cu_seqlens: (N+1,) | None
        split_indices: (NS, 2) | None

    Returns:
        dw1:     (B, T, HQ, K) — gradient w.r.t. w1
        dw2:     (B, T, HQ, K) — gradient w.r.t. w2
        dk_new:  (B, T, HQ, K) — dk propagated through reflector
    """
    B, T, HQ, K = dk.shape
    H = k.shape[2]
    G = HQ // H

    if split_indices is None and cu_seqlens is not None:
        split_indices = _prepare_chunk_indices(cu_seqlens, S)

    if cu_seqlens is None:
        N = B
        NS = N * triton.cdiv(T, S)
        chunk_offsets_ptr = None
        split_offsets_ptr = None
    else:
        N = len(cu_seqlens) - 1
        chunk_offsets = _prepare_chunk_offsets(cu_seqlens, BT)
        split_offsets = _prepare_chunk_offsets(cu_seqlens, S)
        chunk_offsets_ptr = chunk_offsets
        split_offsets_ptr = split_offsets
        NS = int(split_offsets[-1].item())

    grid = (NS, HQ)
    dw1 = torch.empty_like(dk, dtype=torch.float32)
    dw2 = torch.empty_like(dk, dtype=torch.float32)
    dk_new = torch.empty_like(dk, dtype=torch.float32)

    num_warps = 8 if K == 128 else 4

    _chunk_cumprod_householder_bwd_kernel[grid](
        hc_suffix, dhc_whole,
        k, dk, w1, w2, dw1, dw2, dk_new,
        cu_seqlens, split_indices,
        chunk_offsets_ptr, split_offsets_ptr,
        BT=BT, K=K, BK=K, T=T, S=S, G=G, H=H, HQ=HQ,
        IS_VARLEN=(cu_seqlens is not None),
        num_warps=num_warps,
        num_stages=2,
    )
    return dw1, dw2, dk_new


# ============================================================================
# PyTorch Reference Implementation
# ============================================================================


def chunk_cumprod_householder_fwd_ref(
    k: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    S: int,
    BT: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure PyTorch reference for the forward pass."""
    B, T, H, K = k.shape
    dtype = k.dtype
    device = k.device
    NS_total = B * ((T + S - 1) // S)
    NT_total = B * ((T + BT - 1) // BT)

    k_new = torch.zeros_like(k)
    hc_suffix = torch.zeros(NT_total, H, K, K, device=device, dtype=torch.float32)
    hc_whole = torch.zeros(NS_total, H, K, K, device=device, dtype=torch.float32)
    k_f = k.float()
    w1_f = w1.float()
    w2_f = w2.float()

    for b in range(B):
        for h in range(H):
            for i_s in range(NS_total // B):
                # Each split is independent: h starts from zero
                h_state = torch.zeros(K, K, device=device, dtype=torch.float32)
                NT_small = (min(S, T - i_s * S) + BT - 1) // BT

                for i_t_small in range(NT_small - 1, -1, -1):
                    t_s = i_s * S + i_t_small * BT
                    t_e = min(T, t_s + BT)
                    global_idx = b * (NT_total // B) + t_s // BT

                    hc_suffix[global_idx, h] = h_state

                    k_t = k_f[b, t_s:t_e, h]  # [BT', K]
                    w1_t = w1_f[b, t_s:t_e, h]  # [BT', K]
                    w2_t = w2_f[b, t_s:t_e, h]  # [BT', K]

                    k_new[b, t_s:t_e, h] = (k_t - k_t @ h_state.T).to(dtype)
                    v_new = w1_t - (h_state @ w1_t.T).T  # [BT', K]
                    h_state += v_new.T @ w2_t  # [K, BT'] @ [BT', K]

                hc_whole[b * (NS_total // B) + i_s, h] = h_state

    return k_new, hc_suffix.to(dtype), hc_whole.to(dtype)


def chunk_cumprod_householder_bwd_ref(
    w1: torch.Tensor,
    w2: torch.Tensor,
    hc_suffix: torch.Tensor,
    dhc_whole: torch.Tensor,
    k: torch.Tensor,
    dk: torch.Tensor,
    S: int,
    BT: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure PyTorch reference for the backward pass."""
    B, T, HQ, K = dk.shape
    H = k.shape[2]
    G = HQ // H
    device = k.device
    NS_total = B * ((T + S - 1) // S)
    NT_total = B * ((T + BT - 1) // BT)

    dw1 = torch.zeros_like(dk, dtype=torch.float32)
    dw2 = torch.zeros_like(dk, dtype=torch.float32)
    dk_new = torch.zeros_like(dk, dtype=torch.float32)

    k_f = k.float()
    dk_f = dk.float()
    w1_f = w1.float()
    w2_f = w2.float()
    hc_f = hc_suffix.float()
    dhc_whole_f = dhc_whole.float()

    for b in range(B):
        for hq in range(HQ):
            h = hq // G  # GQA mapping
            for i_s in range(NS_total // B):
                # Each split is independent in forward, so dhc starts fresh
                dhc = dhc_whole_f[b * (NS_total // B) + i_s, hq].clone()  # [K, K]
                NT_small = (min(S, T - i_s * S) + BT - 1) // BT
                h_state = torch.zeros(K, K, device=device, dtype=torch.float32)

                for i_t_small in range(0, NT_small):
                    t_s = i_s * S + i_t_small * BT
                    t_e = min(T, t_s + BT)
                    global_idx = b * (NT_total // B) + t_s // BT

                    k_t = k_f[b, t_s:t_e, h]  # [BT', K]
                    dk_t = dk_f[b, t_s:t_e, hq]  # [BT', K]
                    hc_t = hc_f[global_idx, h]  # [K, K]

                    # dk_new = dk - dk @ hc
                    dk_new[b, t_s:t_e, hq] = dk_t - dk_t @ hc_t

                    w1_t = w1_f[b, t_s:t_e, h]  # [BT', K]
                    w2_t = w2_f[b, t_s:t_e, h]  # [BT', K]

                    # dh = dhc - hc^T @ dhc
                    dh = dhc - hc_t.T @ dhc

                    # dw2 = w1 @ dh, dw1 = w2 @ dh^T
                    dw2[b, t_s:t_e, hq] = w1_t @ dh
                    dw1[b, t_s:t_e, hq] = w2_t @ dh.T

                    # Update dhc for next iteration
                    dhc = dhc - (dhc @ w2_t.T) @ w1_t
                    dhc = dhc - dk_t.T @ k_t

    return dw1, dw2, dk_new


# ============================================================================
# Gradient Check Helper
# ============================================================================


def _check_gradients(
    k: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    S: int,
    BT: int,
    eps: float = 1e-2,
) -> dict:
    """Finite-difference gradient check."""
    torch.manual_seed(42)
    B, T, H, K = k.shape

    # Forward
    k_new, hc_suffix, hc_whole = chunk_cumprod_householder_fwd(
        k=k, w1=w1, w2=w2, S=S, BT=BT,
    )

    # Random output gradient (scalar loss for simplicity)
    loss = k_new.sum() + hc_suffix.sum() + hc_whole.sum()

    # Autograd reference
    k_ref = k.detach().clone().requires_grad_(True)
    w1_ref = w1.detach().clone().requires_grad_(True)
    w2_ref = w2.detach().clone().requires_grad_(True)

    k_new_ref, hc_suffix_ref, hc_whole_ref = chunk_cumprod_householder_fwd_ref(
        k=k_ref, w1=w1_ref, w2=w2_ref, S=S, BT=BT,
    )
    loss_ref = k_new_ref.sum() + hc_suffix_ref.sum() + hc_whole_ref.sum()
    loss_ref.backward()

    # Backward with random upstream gradients
    dk_up = torch.randn_like(k_new) * 0.01
    dhc_whole_up = torch.randn_like(hc_whole) * 0.01

    dw1_t, dw2_t, dk_new_t = chunk_cumprod_householder_bwd(
        w1=w1, w2=w2, hc_suffix=hc_suffix, dhc_whole=dhc_whole_up,
        k=k, dk=dk_up, S=S, BT=BT,
    )

    # Compare with reference backward
    dw1_ref_bwd, dw2_ref_bwd, dk_new_ref_bwd = chunk_cumprod_householder_bwd_ref(
        w1=w1, w2=w2, hc_suffix=hc_suffix, dhc_whole=dhc_whole_up,
        k=k, dk=dk_up, S=S, BT=BT,
    )

    results = {
        "k_new_T_vs_ref": (k_new - k_new_ref.detach()).abs().max().item(),
        "hc_whole_T_vs_ref": (hc_whole - hc_whole_ref.detach()).abs().max().item(),
        "dw1_T_vs_refB": (dw1_t - dw1_ref_bwd).abs().max().item(),
        "dw2_T_vs_refB": (dw2_t - dw2_ref_bwd).abs().max().item(),
        "dk_new_T_vs_refB": (dk_new_t - dk_new_ref_bwd).abs().max().item(),
    }
    return results


# ============================================================================
# Correctness Tests
# ============================================================================


def _test_correctness():
    """Verify Triton kernels against PyTorch references."""
    print("=" * 70)
    print("Correctness: Chunk Cumprod Householder")
    print("=" * 70)

    test_configs = [
        # (B, T, H, K, S, BT)
        (1, 64, 2, 32, 32, 16),
        (1, 128, 2, 64, 64, 32),
        (2, 64, 4, 64, 32, 16),
        (1, 128, 2, 128, 64, 32),  # K=128 → 8 warps
    ]

    torch.manual_seed(42)
    all_pass = True

    for B, T_d, H_d, K_d, S_d, BT_d in test_configs:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device = "cuda" if torch.cuda.is_available() else "cpu"

        k = torch.randn(B, T_d, H_d, K_d, device=device, dtype=dtype) * 0.1
        w1 = torch.randn(B, T_d, H_d, K_d, device=device, dtype=dtype) * 0.1
        w2 = torch.randn(B, T_d, H_d, K_d, device=device, dtype=dtype) * 0.1

        if device == "cuda":
            # Forward
            k_new_t, hc_suffix_t, hc_whole_t = chunk_cumprod_householder_fwd(
                k=k, w1=w1, w2=w2, S=S_d, BT=BT_d,
            )

        k_new_r, hc_suffix_r, hc_whole_r = chunk_cumprod_householder_fwd_ref(
            k=k, w1=w1, w2=w2, S=S_d, BT=BT_d,
        )

        if device == "cuda":
            # Backward
            HQ = H_d
            dk_up = torch.randn(B, T_d, HQ, K_d, device=device, dtype=torch.float32) * 0.01
            dhc_up = torch.randn_like(hc_whole_t, dtype=torch.float32) * 0.01

            dw1_t, dw2_t, dk_new_t = chunk_cumprod_householder_bwd(
                w1=w1, w2=w2, hc_suffix=hc_suffix_t, dhc_whole=dhc_up,
                k=k, dk=dk_up, S=S_d, BT=BT_d,
            )

            dw1_r, dw2_r, dk_new_r = chunk_cumprod_householder_bwd_ref(
                w1=w1, w2=w2, hc_suffix=hc_suffix_t, dhc_whole=dhc_up,
                k=k, dk=dk_up, S=S_d, BT=BT_d,
            )

            diffs = {
                "k_new": (k_new_t.float() - k_new_r.float()).abs().max().item(),
                "hc_whole": (hc_whole_t.float() - hc_whole_r.float()).abs().max().item(),
                "dw1": (dw1_t - dw1_r).abs().max().item(),
                "dw2": (dw2_t - dw2_r).abs().max().item(),
                "dk_new": (dk_new_t - dk_new_r).abs().max().item(),
            }
            max_diff = max(diffs.values())
            ok = not math.isnan(max_diff) and max_diff < 0.1
            status = "✅" if ok else "❌"
            if not ok:
                all_pass = False
            parts = " ".join(f"{k}={v:.2e}" for k, v in diffs.items())
            print(f"  B={B} T={T_d} H={H_d} K={K_d} S={S_d} BT={BT_d}: {parts} {status}")
        else:
            print(f"  [CPU] ref shapes: k_new={k_new_r.shape} (skip Triton on CPU)")

    if all_pass:
        print("\n  All correctness tests passed! ✅")
    else:
        print("\n  Some tests failed! ❌")
    print()


# ============================================================================
# Benchmark
# ============================================================================


def _benchmark():
    """Benchmark Triton vs PyTorch reference."""
    if not torch.cuda.is_available():
        print("  [SKIP] No CUDA GPU available for benchmarking.")
        return

    print("=" * 70)
    print("Benchmark: Chunk Cumprod Householder")
    print("=" * 70)

    B, T_d, H_d, K_d = 2, 256, 4, 64
    S_d, BT_d = 64, 16
    dtype = torch.float16
    device = "cuda"

    torch.manual_seed(42)
    k = torch.randn(B, T_d, H_d, K_d, device=device, dtype=dtype) * 0.1
    w1 = torch.randn(B, T_d, H_d, K_d, device=device, dtype=dtype) * 0.1
    w2 = torch.randn(B, T_d, H_d, K_d, device=device, dtype=dtype) * 0.1

    # Warmup
    for _ in range(10):
        chunk_cumprod_householder_fwd(k=k, w1=w1, w2=w2, S=S_d, BT=BT_d)
    torch.cuda.synchronize()

    ms_fwd_T = do_bench(lambda: chunk_cumprod_householder_fwd(
        k=k, w1=w1, w2=w2, S=S_d, BT=BT_d))
    ms_fwd_R = do_bench(lambda: chunk_cumprod_householder_fwd_ref(
        k=k, w1=w1, w2=w2, S=S_d, BT=BT_d))

    # Backward benchmark
    k_new_t, hc_suffix_t, hc_whole_t = chunk_cumprod_householder_fwd(
        k=k, w1=w1, w2=w2, S=S_d, BT=BT_d)
    dk_up = torch.randn(B, T_d, H_d, K_d, device=device, dtype=torch.float32) * 0.01
    dhc_up = torch.randn_like(hc_whole_t, dtype=torch.float32) * 0.01

    for _ in range(10):
        chunk_cumprod_householder_bwd(
            w1=w1, w2=w2, hc_suffix=hc_suffix_t, dhc_whole=dhc_up,
            k=k, dk=dk_up, S=S_d, BT=BT_d)
    torch.cuda.synchronize()

    ms_bwd_T = do_bench(lambda: chunk_cumprod_householder_bwd(
        w1=w1, w2=w2, hc_suffix=hc_suffix_t, dhc_whole=dhc_up,
        k=k, dk=dk_up, S=S_d, BT=BT_d))
    ms_bwd_R = do_bench(lambda: chunk_cumprod_householder_bwd_ref(
        w1=w1, w2=w2, hc_suffix=hc_suffix_t, dhc_whole=dhc_up,
        k=k, dk=dk_up, S=S_d, BT=BT_d))

    print(f"\n  Shape: B={B} T={T_d} H={H_d} K={K_d} S={S_d} BT={BT_d}")
    print(f"  Forward:  Triton={ms_fwd_T:.4f}ms  Ref={ms_fwd_R:.4f}ms  "
          f"Speedup={ms_fwd_R/ms_fwd_T:.1f}x")
    print(f"  Backward: Triton={ms_bwd_T:.4f}ms  Ref={ms_bwd_R:.4f}ms  "
          f"Speedup={ms_bwd_R/ms_bwd_T:.1f}x")
    print()


# ============================================================================
# Main
# ============================================================================


def main():
    print()
    _test_correctness()
    _benchmark()


# ============================================================================
# PERFORMANCE NOTES
# ============================================================================
#
# 1. Algorithm Characteristics
#    - Computes the cumulative product of Householder reflectors.
#    - Each reflector: R_t = I - w1_t @ w2_t^T (rank-BT update to identity).
#    - Cumulative product: H_suffix[t] = Π_{i=t}^{end} (I - w1_i @ w2_i^T).
#    - k_new[t] = k[t] @ H_suffix[t]^T (removes "future" components).
#    - This is the core orthogonalization step in the Householder GDN variant.
#
# 2. Two-level Tiling
#    - Split level (S): coarse-grained grouping of the sequence.
#    - Small chunk level (BT): fine-grained blocks within a split.
#    - h accumulates across small chunks within a split.
#    - hc_suffix stores intermediate h for backward pass.
#    - hc_whole stores final h for inter-split communication.
#
# 3. Memory Access Pattern
#    - h (K×K matrix) lives in registers throughout the split loop.
#    - Each small chunk: load k[BT,K], w1[BT,K], w2[BT,K].
#    - Store k_new[BT,K] and hc_suffix[K,K] per small chunk.
#    - All accesses use block_ptr with boundary_check.
#    - Bandwidth: O(K²) intermediate state, O(BT·K) per small chunk.
#
# 4. Compute Characteristics
#    - Per small chunk: 3 matrix multiplies (k@h^T, h@w1, v_new@w2).
#    - All use tl.dot → Tensor Core MMA instructions.
#    - BK = K (full K dimension in one tile) — good register reuse.
#    - Arithmetic intensity: O(BT·K²) / O(BT·K) = O(K) FLOP/byte.
#    - For K ≥ 64: compute-bound. For K < 64: memory-bound.
#
# 5. Backward Algorithm
#    - Reverse-mode differentiation of the forward recurrence.
#    - Each step computes: dk_new, dw1, dw2, and updates dhc.
#    - GQA support: G = HQ // H allows different numbers of query/key heads.
#    - Gradient accumulation in fp32 for numerical stability.
#
# 6. Optimization Details
#    - K=128 uses 8 warps (empirically needed for correctness).
#    - BK = K (fully unrolled K dimension in one tile).
#    - h stored entirely in registers — no shared memory needed.
#    - No autotune: BK is fixed to K, grid is determined by split count.
#
# 7. References
#    - fla-org/flash-linear-attention: chunk_cumprod_householder_fwd/bwd_kernel
#    - "Householder Transformer" — Householder reflections for attention
#    - "GateLoop" / "GDN" — gated delta rule with Householder orthogonalization


if __name__ == "__main__":
    main()
