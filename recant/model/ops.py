"""Token-mixing kernels with swappable backends: gated delta rule (GDN) and softmax attention.

GDN backends: `torch` (chunked reference ported from the HF Qwen3.5 fallback, differentiable,
runs anywhere) and `fla` (flash-linear-attention Triton kernels). Attention uses torch SDPA;
the cached (no-grad) path processes queries in blocks with an explicit offset causal mask.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .. import prof

_state = {"gdn": "torch", "attn_gqa": True, "cached": "masked", "restrict_sdpa": False}


def set_gdn_backend(name: str) -> None:
    if name not in GDN_BACKENDS:
        raise KeyError(f"unknown GDN backend {name!r}; have {list(GDN_BACKENDS)}")
    _state["gdn"] = name


def get_gdn_backend() -> str:
    return _state["gdn"]


def set_attn_gqa(flag: bool) -> None:
    _state["attn_gqa"] = bool(flag)


def _l2norm(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def gdn_torch(q, k, v, g, beta, initial_state=None, output_final_state=False, chunk_size=64):
    """Gated delta rule. q,k: [B,T,H,Dk] (already repeated to H value heads), v: [B,T,H,Dv],
    g,beta: [B,T,H] (g = log decay <= 0). Returns ([B,T,H,Dv], final state [B,H,Dk,Dv] | None).
    q and k are L2-normalized here and q is scaled by Dk^-0.5 (fla semantics)."""
    in_dtype = q.dtype
    b, t, h, dk = k.shape
    dv = v.shape[-1]
    q, k, v, beta, g = [x.transpose(1, 2).to(torch.float32).contiguous() for x in (q, k, v, beta, g)]
    q, k = _l2norm(q), _l2norm(k)
    q = q * dk ** -0.5
    pad = (chunk_size - t % chunk_size) % chunk_size
    q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
    beta, g = (F.pad(x, (0, pad)) for x in (beta, g))
    n = (t + pad) // chunk_size
    v_beta, k_beta = v * beta.unsqueeze(-1), k * beta.unsqueeze(-1)
    q, k, k_beta, v_beta = [x.reshape(b, h, n, chunk_size, x.shape[-1]) for x in (q, k, k_beta, v_beta)]
    g = g.reshape(b, h, n, chunk_size)
    upper = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device).triu(1)
    cum = g.cumsum(-1)
    pair = (cum.unsqueeze(-1) - cum.unsqueeze(-2)).masked_fill(upper, float("-inf")).exp()
    ut = (k_beta @ k.transpose(-1, -2)) * pair
    intra = (q @ k.transpose(-1, -2)) * pair
    dk_beta = k_beta * cum.exp().unsqueeze(-1)
    new_v = torch.linalg.solve_triangular(ut, v_beta, upper=False, unitriangular=True)
    k_cum = torch.linalg.solve_triangular(ut, dk_beta, upper=False, unitriangular=True)
    state = (torch.zeros(b, h, dk, dv, dtype=torch.float32, device=q.device) if initial_state is None
             else initial_state.to(torch.float32))
    out = torch.zeros_like(new_v)
    q = q * cum.exp().unsqueeze(-1)
    k = k * (cum[..., -1:] - cum).exp().unsqueeze(-1)
    chunk_decay = cum[..., -1].exp()[..., None, None]
    outs = []
    for i in range(n):
        v_new = new_v[:, :, i] - k_cum[:, :, i] @ state
        outs.append(q[:, :, i] @ state + intra[:, :, i] @ v_new)
        state = state * chunk_decay[:, :, i] + k[:, :, i].transpose(-1, -2) @ v_new
    out = torch.stack(outs, 2).reshape(b, h, n * chunk_size, dv)[:, :, :t]
    return out.transpose(1, 2).contiguous().to(in_dtype), (state if output_final_state else None)


def gdn_fla(q, k, v, g, beta, initial_state=None, output_final_state=False):
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    o, s = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=initial_state,
                                  output_final_state=output_final_state, use_qk_l2norm_in_kernel=True)
    return o, s


GDN_BACKENDS = {"torch": gdn_torch, "fla": gdn_fla}


def gdn_core(q, k, v, g, beta, initial_state=None, output_final_state=False):
    with prof.rf("b:gdn"):
        return _gdn_core(q, k, v, g, beta, initial_state, output_final_state)


def _gdn_core(q, k, v, g, beta, initial_state, output_final_state):
    return GDN_BACKENDS[_state["gdn"]](q, k, v, g, beta, initial_state=initial_state,
                                       output_final_state=output_final_state)


# --------------------------------------------------------------------------- attention
def set_cached_impl(name: str) -> None:
    if name not in ("masked", "flash_lse"):
        raise KeyError(name)
    _state["cached"] = name


def set_restrict_sdpa(flag: bool) -> None:
    """Allow only flash / cuDNN / mem-efficient SDPA kernels (never the O(T^2) math kernel) on CUDA."""
    _state["restrict_sdpa"] = bool(flag)


def _sdpa(q, k, v, **kw):
    if _state["restrict_sdpa"] and q.is_cuda:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        try:
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                return _sdpa_inner(q, k, v, **kw)
        except RuntimeError:
            pass                      # no allowed kernel supports this call: fall back to the default dispatch
    return _sdpa_inner(q, k, v, **kw)


def _sdpa_inner(q, k, v, **kw):
    scale = q.shape[-1] ** -0.5
    if q.shape[1] != k.shape[1]:
        if _state["attn_gqa"]:
            try:
                return F.scaled_dot_product_attention(q, k, v, scale=scale, enable_gqa=True, **kw)
            except (RuntimeError, TypeError):
                _state["attn_gqa"] = False
        rep = q.shape[1] // k.shape[1]
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
    return F.scaled_dot_product_attention(q, k, v, scale=scale, **kw)


def attn_full(q, k, v):
    """Causal attention, Tq == Tk. q [B,Hq,T,D], k/v [B,Hkv,T,D]."""
    with prof.rf("b:attention"):
        return _sdpa(q, k, v, is_causal=True)


SCORE_BUDGET_BYTES = 1 << 31   # cap on one block's fp32 score matrix in case SDPA falls back to the math kernel


def _attn_lse_ref(q, k, v, causal):
    """Reference attention that also returns the log-sum-exp. q [B,H,Tq,D], k/v [B,H,Tk,D] (same head count).
    causal = square lower-triangular (Tq == Tk)."""
    scale = q.shape[-1] ** -0.5
    sc = (q.float() @ k.float().transpose(-1, -2)) * scale
    if causal:
        sc = sc.masked_fill(torch.ones(sc.shape[-2:], dtype=torch.bool, device=sc.device).triu(1), float("-inf"))
    lse = torch.logsumexp(sc, -1)
    return (torch.softmax(sc, -1) @ v.float()).to(q.dtype), lse


def _attn_lse_flash(q, k, v, causal):
    """torch's flash-attention forward with its log-sum-exp output (CUDA only; used on the no-grad path)."""
    out, lse = torch.ops.aten._scaled_dot_product_flash_attention(q, k, v, 0.0, causal, False,
                                                                  scale=q.shape[-1] ** -0.5)[:2]
    return out, lse


def merge_attention(o1, l1, o2, l2):
    """Combine two attention results over disjoint key sets from their outputs and log-sum-exps."""
    lse = torch.logaddexp(l1, l2)
    w1, w2 = torch.exp(l1 - lse)[..., None], torch.exp(l2 - lse)[..., None]
    return (o1.float() * w1 + o2.float() * w2).to(o1.dtype)


def attn_cached_two_part(q, k, v, lse_fn):
    """Offset-causal attention as (prefix keys, no mask) + (own keys, causal), merged through the LSE. No mask
    means the flash kernel applies to both parts. q [B,Hq,Tq,D], k/v [B,Hkv,Tk,D], Tk = prefix + Tq."""
    tq, tk = q.shape[2], k.shape[2]
    prefix = tk - tq
    if q.shape[1] != k.shape[1]:
        rep = q.shape[1] // k.shape[1]
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
    o2, l2 = lse_fn(q, k[:, :, prefix:], v[:, :, prefix:], True)
    if prefix == 0:
        return o2
    o1, l1 = lse_fn(q, k[:, :, :prefix], v[:, :, :prefix], False)
    return merge_attention(o1, l1, o2, l2)


def attn_cached_masked(q, k, v, block: int = 2048):
    """Causal attention for Tq new queries over Tk = prefix + Tq keys (no grad path).
    Query i may see keys j <= prefix + i. Query blocks are sized so a block's fp32 scores stay bounded."""
    tq, tk = q.shape[2], k.shape[2]
    prefix = tk - tq
    if prefix == 0:
        return _sdpa(q, k, v, is_causal=True)
    block = max(64, min(block, SCORE_BUDGET_BYTES // (4 * q.shape[1] * tk)))
    out = torch.empty_like(q)
    for s in range(0, tq, block):
        e = min(s + block, tq)
        i = torch.arange(s, e, device=q.device)[:, None]
        j = torch.arange(prefix + e, device=q.device)[None, :]
        out[:, :, s:e] = _sdpa(q[:, :, s:e], k[:, :, :prefix + e], v[:, :, :prefix + e], attn_mask=(j <= prefix + i))
    return out


def attn_cached(q, k, v, block: int = 2048):
    with prof.rf("b:attention"):
        if _state["cached"] == "flash_lse" and q.is_cuda:
            return attn_cached_two_part(q, k, v, _attn_lse_flash)
        return attn_cached_masked(q, k, v, block)
