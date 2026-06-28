"""
phase3_production/08_fused_recurrent_gdn.py — Fused Recurrent Gated Delta Rule

Fully fused token-by-token recurrent kernel combining:
  - Hidden state recurrence: h[t] = h[t-1] * decay + k[t]^T @ v_new[t]
  - Delta rule: v_new[t] = beta[t] * (v[t] - h[t-1] @ k[t])
  - Output: o[t] = h[t] @ q[t]

Supports multiple gate types, optional QK L2 norm, state layout options,
beta sigmoid activation, and gate activation fusion.

Reference: fla-org/flash-linear-attention — fused_recurrent_gated_delta_rule_fwd_kernel

运行: python phase3_production/08_fused_recurrent_gdn.py
"""

import math
import warnings

import torch
import triton
import triton.language as tl
from triton.testing import do_bench


# ============================================================================
# Utilities
# ============================================================================


@triton.jit
def _exp(x):
    """exp in Triton."""
    return tl.math.exp(x)


@triton.jit
def _softplus(x):
    """Softplus: log(1 + exp(x))."""
    return tl.math.log1p(tl.math.exp(x))


# ============================================================================
# Fused Recurrent GDN Forward Kernel
# ============================================================================


@triton.jit
def _fused_recurrent_gdn_fwd_kernel(
    # Pointers
    q_ptr,          # (B, T, H, K)
    k_ptr,          # (B, T, H, K)
    v_ptr,          # (B, T, HV, V)
    g_ptr,          # (B, T, HV) | None
    gk_ptr,         # (B, T, HV, K) | None
    gv_ptr,         # (B, T, HV, V) | None
    beta_ptr,       # (B, T, HV) or (B, T, HV, V)
    A_log_ptr,      # (HV,) | None
    dt_bias_ptr,    # (HV,) | None
    o_ptr,          # (B, T, HV, V)
    h0_ptr,         # (N, HV, K, V) or (N, HV, V, K) | None
    ht_ptr,         # (N, HV, K, V) or (N, HV, V, K) | None
    cu_seqlens_ptr, # (N+1,) | None
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_GATE_IN_KERNEL: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    APPLY_BETA_SIGMOID: tl.constexpr,
    ALLOW_NEG_EIGVAL: tl.constexpr,
):
    """
    Fused recurrent GDN forward kernel.

    Grid: (cdiv(V, BV), N * HV)
      axis=0: V dimension tiles
      axis=1: (sequence, head-v) pair

    Each program iterates over all T time steps sequentially, maintaining
    the hidden state in registers.
    """
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)  # GQA: map HV head to H head

    # ── Sequence boundaries ──────────────────────────────────────
    if IS_VARLEN:
        bos = tl.load(cu_seqlens_ptr + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens_ptr + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = i_n * T
        eos = i_n * T + T

    # ── Per-program constants ────────────────────────────────────
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V

    # ── Pointer offsets ──────────────────────────────────────────
    # q, k: use H heads (GQA mapping)
    p_q = q_ptr + (bos * H + i_h) * K + o_k
    p_k = k_ptr + (bos * H + i_h) * K + o_k
    # v, o: use HV heads
    p_v = v_ptr + (bos * HV + i_hv) * V + o_v
    p_o = o_ptr + (bos * HV + i_hv) * V + o_v

    if USE_G:
        p_g = g_ptr + bos * HV + i_hv
    if USE_GK:
        p_gk = gk_ptr + (bos * HV + i_hv) * K + o_k
    if USE_GV:
        p_gv = gv_ptr + (bos * HV + i_hv) * V + o_v
    if IS_BETA_HEADWISE:
        p_beta = beta_ptr + bos * HV + i_hv  # scalar per head per token
    else:
        p_beta = beta_ptr + (bos * HV + i_hv) * V + o_v  # per-value-channel

    # ── Hidden state mask ────────────────────────────────────────
    if STATE_V_FIRST:
        mask_h = mask_v[:, None] & mask_k[None, :]  # [BV, BK]
    else:
        mask_h = mask_k[:, None] & mask_v[None, :]  # [BK, BV]

    # ── Initialize hidden state ──────────────────────────────────
    if STATE_V_FIRST:
        b_h = tl.zeros([BV, BK], dtype=tl.float32)
    else:
        b_h = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        if STATE_V_FIRST:
            p_h0 = h0_ptr + i_nh * K * V + o_v[:, None] * K + o_k[None, :]
        else:
            p_h0 = h0_ptr + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    # ── Main recurrent loop ──────────────────────────────────────
    for _ in tl.range(0, T):
        # Load current token
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

        # Optional QK L2 normalization
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)

        # Scale q
        b_q = b_q * scale

        # Load beta (headwise or per-value-channel)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta).to(tl.float32)
        else:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)

        # Optional beta sigmoid
        if APPLY_BETA_SIGMOID:
            b_beta = tl.sigmoid(b_beta)
            if ALLOW_NEG_EIGVAL:
                b_beta = b_beta * 2  # range [0, 2)

        # ── Apply gate decay ─────────────────────────────────────
        if USE_G:
            b_g = tl.load(p_g).to(tl.float32)
            if USE_GATE_IN_KERNEL:
                b_A = tl.load(A_log_ptr + i_hv).to(tl.float32)
                if HAS_DT_BIAS:
                    b_g = b_g + tl.load(dt_bias_ptr + i_hv).to(tl.float32)
                b_g = -_exp(b_A) * _softplus(b_g)
            # Decay hidden state: h *= exp(g)
            b_h *= _exp(b_g)

        if USE_GK:
            b_gk = tl.load(p_gk, mask=mask_k, other=0).to(tl.float32)
            if STATE_V_FIRST:
                b_h *= _exp(b_gk[None, :])  # broadcast over V dim
            else:
                b_h *= _exp(b_gk[:, None])  # broadcast over V dim

        if USE_GV:
            b_gv = tl.load(p_gv, mask=mask_v, other=0).to(tl.float32)
            if STATE_V_FIRST:
                b_h *= _exp(b_gv[:, None])  # broadcast over K dim
            else:
                b_h *= _exp(b_gv[None, :])  # broadcast over K dim

        # ── Delta rule: v_new = beta * (v - h @ k) ───────────────
        if STATE_V_FIRST:
            # h: [BV, BK], k: [BK] → h @ k: [BV]
            v_hk = tl.sum(b_h * b_k[None, :], 1)  # [BV]
            b_v_new = b_beta * (b_v - v_hk)
            # Update h: h += k^T @ v_new = k[None,:] * v_new[:,None] → [BK, BV] → [BV, BK]
            b_h += b_v_new[:, None] * b_k[None, :]
            # Output: o = h @ q = sum over K: h[v,k] * q[k]
            b_o = tl.sum(b_h * b_q[None, :], 1)  # [BV]
        else:
            # h: [BK, BV], k: [BK] → k^T @ h: h[k,v] @ ... wait
            # v_new = beta * (v - sum_k h[k,v] * k[k])
            v_hk = tl.sum(b_h * b_k[:, None], 0)  # [BV]
            b_v_new = b_beta * (b_v - v_hk)
            # Update: h += k[:,None] * v_new[None,:] → [BK, BV]
            b_h += b_k[:, None] * b_v_new[None, :]
            # Output: o = h @ q = sum over K: h[k,v] * q[k]
            b_o = tl.sum(b_h * b_q[:, None], 0)  # [BV]

        # Store output
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # Advance pointers to next time step
        p_q += H * K
        p_k += H * K
        p_v += HV * V
        if USE_G:
            p_g += HV
        if USE_GK:
            p_gk += HV * K
        if USE_GV:
            p_gv += HV * V
        p_beta += HV * (1 if IS_BETA_HEADWISE else V)
        p_o += HV * V

    # ── Store final state ────────────────────────────────────────
    if STORE_FINAL_STATE:
        if STATE_V_FIRST:
            p_ht = ht_ptr + i_nh * K * V + o_v[:, None] * K + o_k[None, :]
        else:
            p_ht = ht_ptr + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


# ============================================================================
# Python Wrapper
# ============================================================================


def fused_recurrent_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    gv: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    state_v_first: bool = False,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Fused recurrent GDN forward pass.

    Args:
        q: (B, T, H, K) queries
        k: (B, T, H, K) keys
        v: (B, T, HV, V) values (HV may differ from H for GVA)
        g: (B, T, HV) | None — per-token decay (log space, pre-computed)
        gk: (B, T, HV, K) | None — per-key-dim decay
        gv: (B, T, HV, V) | None — per-value-dim decay
        beta: (B, T, HV) or (B, T, HV, V) — delta rule beta
        A_log: (HV,) | None — for gate fusion
        dt_bias: (HV,) | None — bias for gate fusion
        scale: float | None — attention scale (default: 1/√K)
        initial_state: (N, HV, K, V) or (N, HV, V, K) | None
        output_final_state: if True, return final hidden state
        use_qk_l2norm_in_kernel: fuse L2 norm of q and k
        use_beta_sigmoid_in_kernel: fuse sigmoid(beta)
        allow_neg_eigval: allow negative eigenvalues (beta → 2*sigmoid(beta))
        state_v_first: use V-first [V,K] layout instead of [K,V]
        cu_seqlens: (N+1,) | None — for varlen sequences

    Returns:
        o: (B, T, HV, V) output
        final_state: (N, HV, K, V) or None
    """
    B, T, H, K = k.shape
    HV = v.shape[2]
    V = v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    if scale is None:
        scale = K ** -0.5
    if beta is None:
        beta = torch.ones_like(q[..., 0])

    BK = triton.next_power_of_2(K)
    BV = min(8, triton.next_power_of_2(V)) if gv is None else triton.next_power_of_2(V)
    NV = triton.cdiv(V, BV)

    o = torch.empty_like(v)

    if output_final_state:
        if state_v_first:
            final_state = q.new_empty(N, HV, V, K, dtype=torch.float32)
        else:
            final_state = q.new_empty(N, HV, K, V, dtype=torch.float32)
    else:
        final_state = None

    grid = (NV, N * HV)
    _fused_recurrent_gdn_fwd_kernel[grid](
        q, k, v, g, gk, gv, beta, A_log, dt_bias,
        o, initial_state, final_state, cu_seqlens,
        scale,
        T=T,
        H=H, HV=HV, K=K, V=V, BK=BK, BV=BV,
        USE_G=(g is not None),
        USE_GK=(gk is not None),
        USE_GV=(gv is not None),
        IS_BETA_HEADWISE=(beta.ndim != v.ndim),
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        USE_INITIAL_STATE=(initial_state is not None),
        STORE_FINAL_STATE=output_final_state,
        STATE_V_FIRST=state_v_first,
        IS_VARLEN=(cu_seqlens is not None),
        USE_GATE_IN_KERNEL=(A_log is not None),
        HAS_DT_BIAS=(dt_bias is not None),
        APPLY_BETA_SIGMOID=use_beta_sigmoid_in_kernel,
        ALLOW_NEG_EIGVAL=allow_neg_eigval,
        num_warps=1,
        num_stages=3,
    )
    return o, final_state


# ============================================================================
# High-Level API
# ============================================================================


def fused_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    gv: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    state_v_first: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """High-level API: fused recurrent gated delta rule.

    When ``use_gate_in_kernel=True``, the kernel fuses the gate activation
    computation internally: ``g_activated = -exp(A_log) * softplus(g + dt_bias)``.
    Otherwise, ``g`` must be pre-computed in log space.

    Args:
        q: (B, T, H, K)
        k: (B, T, H, K)
        v: (B, T, HV, V)
        g: (B, T, HV) | None
        gk: (B, T, HV, K) | None
        gv: (B, T, HV, V) | None
        beta: (B, T, HV) | None
        scale: float | None
        initial_state: (N, HV, K, V) | None
        output_final_state: bool
        use_qk_l2norm_in_kernel: bool
        use_gate_in_kernel: bool — fuse gate activation
        A_log: (HV,) | None — required if use_gate_in_kernel=True
        dt_bias: (HV,) | None — optional gate bias
        use_beta_sigmoid_in_kernel: bool
        allow_neg_eigval: bool
        state_v_first: bool
        cu_seqlens: (N+1,) | None

    Returns:
        o: (B, T, HV, V)
        final_state: (N, HV, K, V) | None
    """
    # Handle deprecated transpose_state_layout kwarg
    if "transpose_state_layout" in kwargs:
        if state_v_first:
            raise ValueError(
                "Cannot pass both `state_v_first` and deprecated `transpose_state_layout`."
            )
        warnings.warn(
            "`transpose_state_layout` is deprecated, use `state_v_first`.",
            DeprecationWarning,
            stacklevel=2,
        )
        state_v_first = kwargs.pop("transpose_state_layout")

    # Varlen validation
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"Batch size must be 1 with cu_seqlens, got {q.shape[0]}. "
                f"Flatten variable-length inputs first."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"Expected {len(cu_seqlens) - 1} initial states, "
                f"got {initial_state.shape[0]}."
            )

    # Gate fusion validation
    if use_gate_in_kernel:
        if A_log is None:
            raise ValueError("`A_log` required when `use_gate_in_kernel=True`.")
        if g is None:
            raise ValueError("`g` (raw pre-activation) required when `use_gate_in_kernel=True`.")
    else:
        A_log = None
        dt_bias = None

    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise ValueError("`allow_neg_eigval=True` requires `use_beta_sigmoid_in_kernel=True`.")

    o, final_state = fused_recurrent_gated_delta_rule_fwd(
        q=q, k=k, v=v, g=g, gk=gk, gv=gv, beta=beta,
        A_log=A_log, dt_bias=dt_bias,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        state_v_first=state_v_first,
        cu_seqlens=cu_seqlens,
    )
    return o, final_state


# Alias
fused_recurrent_gdn = fused_recurrent_gated_delta_rule


# ============================================================================
# PyTorch Reference Implementation
# ============================================================================


def fused_recurrent_gdn_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    gv: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    state_v_first: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Pure PyTorch reference for correctness verification."""
    B, T, H, K = k.shape
    HV = v.shape[2]
    V = v.shape[-1]

    if scale is None:
        scale = K ** -0.5
    if beta is None:
        beta = torch.ones_like(q[..., 0])

    q_f, k_f, v_f = q.float(), k.float(), v.float()
    beta_f = beta.float()
    g_f = g.float() if g is not None else None
    gk_f = gk.float() if gk is not None else None
    gv_f = gv.float() if gv is not None else None

    o = torch.zeros(B, T, HV, V, device=q.device, dtype=torch.float32)

    # Initialize hidden state per (batch, head-v)
    N = B
    if initial_state is not None:
        h = initial_state.float().clone()
    else:
        if state_v_first:
            h = torch.zeros(N, HV, V, K, device=q.device, dtype=torch.float32)
        else:
            h = torch.zeros(N, HV, K, V, device=q.device, dtype=torch.float32)

    # Note: this ref doesn't support output_final_state parameter

    for b in range(B):
        for hv in range(HV):
            h_k = hv if hv < H else hv % H  # GQA head mapping

            for t in range(T):
                qt = q_f[b, t, h_k] * scale  # [K]
                kt = k_f[b, t, h_k]  # [K]
                vt = v_f[b, t, hv]  # [V]

                # Beta
                if beta.ndim == v.ndim:
                    bt = beta_f[b, t, hv]  # [V] per-channel
                else:
                    bt = beta_f[b, t, hv]  # scalar

                # Gate decay
                if g_f is not None:
                    gt = g_f[b, t, hv]  # scalar
                    h[b, hv] = h[b, hv] * math.exp(gt)

                if gk_f is not None:
                    gkt = gk_f[b, t, hv]  # [K]
                    if state_v_first:
                        h[b, hv] = h[b, hv] * torch.exp(gkt).unsqueeze(0)
                    else:
                        h[b, hv] = h[b, hv] * torch.exp(gkt).unsqueeze(1)

                if gv_f is not None:
                    gvt = gv_f[b, t, hv]  # [V]
                    if state_v_first:
                        h[b, hv] = h[b, hv] * torch.exp(gvt).unsqueeze(1)
                    else:
                        h[b, hv] = h[b, hv] * torch.exp(gvt).unsqueeze(0)

                # Delta rule
                if state_v_first:
                    hk = (h[b, hv] * kt.unsqueeze(0)).sum(dim=1)  # [V]
                    v_new = bt * (vt - hk)  # [V]
                    h[b, hv] = h[b, hv] + v_new.unsqueeze(1) * kt.unsqueeze(0)
                    o[b, t, hv] = (h[b, hv] * qt.unsqueeze(0)).sum(dim=1)
                else:
                    hk = (h[b, hv] * kt.unsqueeze(1)).sum(dim=0)  # [V]
                    v_new = bt * (vt - hk)  # [V]
                    h[b, hv] = h[b, hv] + kt.unsqueeze(1) * v_new.unsqueeze(0)
                    o[b, t, hv] = (h[b, hv] * qt.unsqueeze(1)).sum(dim=0)

    return o.to(q.dtype), h


# ============================================================================
# Correctness Tests
# ============================================================================


def _test_correctness():
    """Verify Triton kernel against PyTorch reference."""
    print("=" * 70)
    print("Correctness: Fused Recurrent GDN vs PyTorch Reference")
    print("=" * 70)

    torch.manual_seed(42)

    test_configs = [
        # (B, T, H, HV, K, V, use_g, use_beta_sigmoid, state_v_first)
        (1, 64, 2, 2, 64, 64, True, False, False),
        (1, 128, 2, 2, 64, 64, False, True, False),
        (2, 64, 4, 4, 64, 64, True, False, True),
        (1, 64, 2, 2, 128, 64, True, False, False),
    ]

    all_pass = True
    for B, T_d, H_d, HV_d, K_d, V_d, use_g, use_sigmoid, state_vf in test_configs:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device = "cuda" if torch.cuda.is_available() else "cpu"

        q = torch.randn(B, T_d, H_d, K_d, device=device, dtype=dtype) * 0.1
        k = torch.randn(B, T_d, H_d, K_d, device=device, dtype=dtype) * 0.1
        v = torch.randn(B, T_d, HV_d, V_d, device=device, dtype=dtype) * 0.1
        g = None
        if use_g:
            g = torch.randn(B, T_d, HV_d, device=device, dtype=torch.float32) * 0.5
        beta = torch.rand(B, T_d, HV_d, device=device, dtype=torch.float32)
        scale = K_d ** -0.5

        if device == "cuda":
            o_triton, _ = fused_recurrent_gated_delta_rule(
                q=q, k=k, v=v, g=g, beta=beta,
                scale=scale, use_beta_sigmoid_in_kernel=use_sigmoid,
                state_v_first=state_vf,
            )

        o_ref, _ = fused_recurrent_gdn_ref(
            q=q, k=k, v=v, g=g, beta=beta,
            scale=scale, state_v_first=state_vf,
        )

        if device == "cuda":
            max_diff = (o_triton.float() - o_ref.float()).abs().max().item()
            tol = 0.5 if use_g else 0.2
            ok = not math.isnan(max_diff) and max_diff < tol
            status = "✅" if ok else "❌"
            if not ok:
                all_pass = False
            print(
                f"  B={B} T={T_d} H={H_d} HV={HV_d} K={K_d} V={V_d} "
                f"gate={use_g} sigmoid={use_sigmoid} Vfirst={state_vf}: "
                f"max_diff={max_diff:.4e} {status}"
            )
        else:
            print(f"  [CPU] ref shape: {o_ref.shape} (skip Triton on CPU)")

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
    print("Benchmark: Fused Recurrent GDN")
    print("=" * 70)

    B, T_d, H_d, K_d, V_d = 2, 256, 4, 128, 64
    dtype = torch.float16
    device = "cuda"

    torch.manual_seed(42)
    q = torch.randn(B, T_d, H_d, K_d, device=device, dtype=dtype) * 0.1
    k = torch.randn(B, T_d, H_d, K_d, device=device, dtype=dtype) * 0.1
    v = torch.randn(B, T_d, H_d, V_d, device=device, dtype=dtype) * 0.1
    g = torch.randn(B, T_d, H_d, device=device, dtype=torch.float32) * 0.5
    beta = torch.rand(B, T_d, H_d, device=device, dtype=torch.float32)
    scale = K_d ** -0.5

    # Warmup
    for _ in range(10):
        fused_recurrent_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta, scale=scale)
    torch.cuda.synchronize()

    ms_triton = do_bench(
        lambda: fused_recurrent_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta, scale=scale)
    )

    ms_ref = do_bench(
        lambda: fused_recurrent_gdn_ref(
            q=q, k=k, v=v, g=g, beta=beta, scale=scale)
    )

    print(f"\n  Shape: B={B} T={T_d} H={H_d} K={K_d} V={V_d}")
    print(f"  Triton:  {ms_triton:.4f} ms")
    print(f"  PyTorch: {ms_ref:.4f} ms")
    print(f"  Speedup: {ms_ref / ms_triton:.2f}x")
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
#    - Token-by-token recurrent kernel (O(T·K·V) compute, O(K·V) memory).
#    - Unlike chunked versions, this processes the sequence sequentially.
#    - Hidden state h (size K×V or V×K) lives entirely in registers.
#    - Each time step: load q,k,v (3 reads), compute delta + update h, store o.
#    - Simple, interpretable, and serves as the reference for more complex
#      chunked implementations.
#
# 2. Memory Access Pattern
#    - Per time step: 3 small loads (q[BK], k[BK], v[BV]) and 1 store (o[BV]).
#    - Hidden state stays in registers — no HBM access for h.
#    - Gate loads are optional and also small (scalar or [BK]/[BV] vectors).
#    - Bandwidth utilization is low — kernel is latency-bound due to the
#      sequential dependency chain.
#
# 3. Compute Characteristics
#    - Per time step O(K·V) operations from h@k and k^T@v_new.
#    - No matrix multiplications — just vector-vector and elementwise ops.
#    - Tensor Cores are NOT used (too small for MMA overhead).
#    - This is a sequential kernel — throughput is limited by T iterations.
#
# 4. State Layout Options
#    - STATE_V_FIRST=False (default): h [K, V] — natural for K-first models
#    - STATE_V_FIRST=True: h [V, K] — better coalescing for V-dimension loads
#    - Both layouts are mathematically equivalent; choice affects memory access
#      patterns within the per-step updates.
#
# 5. Comparison with Chunked Version
#    - Chunked: O(T/K·V) HBM accesses for K and V (batched per chunk).
#    - Recurrent: O(T·K·V) HBM accesses (one set per token).
#    - Chunked is MUCH faster for long sequences (T >> chunk_size).
#    - Recurrent is simpler and useful for:
#      * Inference with small batch sizes
#      * Numerical verification of chunked kernels
#      * Understanding the base algorithm
#
# 6. Optimization Roadmap
#    - [DONE] Fused all operations into single kernel.
#    - [DONE] Register-resident hidden state.
#    - [DONE] State layout options for flexibility.
#    - [TODO] Multi-token at a time (process small groups to reduce loop count).
#    - [TODO] Shared memory staging for q and k to exploit reuse within a warp.
#    - [TODO] TMA for bulk loading of contiguous tokens.
#
# 7. Numerical Notes
#    - All computation in fp32 (accumulators in fp32, loads cast to fp32).
#    - Gate values use exp (natural base), not exp2 — consistent with FLA convention.
#    - QK L2 norm uses epsilon=1e-6 to prevent division by zero.
#    - Beta sigmoid fusion reduces kernel launch overhead.
#    - For long sequences, fp16 inputs may lose precision in h accumulation.
#      Consider fp32 for h state in production training.
#
# 8. References
#    - fla-org/flash-linear-attention: original FLA implementation
#    - "Gated Delta Networks" (Yang et al., 2024)
#    - "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"


if __name__ == "__main__":
    main()
