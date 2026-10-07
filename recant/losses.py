from __future__ import annotations
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from . import prof

def _ck(fn, *args):
    if torch.is_grad_enabled() and any((isinstance(a, torch.Tensor) and a.requires_grad for a in args)):
        return checkpoint(fn, *args, use_reentrant=False)
    return fn(*args)

def _ce_chunk(h, t, w):
    return F.cross_entropy(F.linear(h, w).float(), t, reduction='sum')

def _ce_sum_impl(h, targets, w, chunk=512):
    total = h.new_zeros((), dtype=torch.float32)
    for s in range(0, h.shape[0], chunk):
        total = total + _ck(_ce_chunk, h[s:s + chunk], targets[s:s + chunk], w)
    return total

def _kl_packed(logits, ids, lp, tail):
    lse = torch.logsumexp(logits, -1)
    lq = logits.gather(1, ids) - lse[:, None]
    kl_top = (lp.exp() * (lp - lq)).sum(-1)
    q_tail = (1.0 - lq.exp().sum(-1)).clamp(min=1e-12)
    kl_tail = tail.exp() * (tail - q_tail.log())
    return kl_top + kl_tail

def _corr_chunk(h_sg, delta, g, w, ids, lp, tail):
    with torch.no_grad():
        z1 = F.linear(h_sg, w).float()
    z2 = F.linear(delta.to(w.dtype), w).float()
    return _kl_packed(z1 + g[:, None].float() * z2, ids, lp, tail).sum()

def _corr_sum_impl(h_sg, delta, g, w, ids, lp, tail, chunk=256):
    total = h_sg.new_zeros((), dtype=torch.float32)
    for s in range(0, h_sg.shape[0], chunk):
        sl = slice(s, s + chunk)
        total = total + _ck(_corr_chunk, h_sg[sl], delta[sl], g[sl], w, ids[sl], lp[sl], tail[sl])
    return total

def _keep_chunk(h_sg, delta, g, w):
    with torch.no_grad():
        z1 = F.linear(h_sg, w).float()
        p = torch.softmax(z1, -1)
        lse1 = torch.logsumexp(z1, -1)
    z2 = F.linear(delta.to(w.dtype), w).float()
    gg = g[:, None].float()
    lse2 = torch.logsumexp(z1 + gg * z2, -1)
    return (lse2 - lse1 - g.float() * (p * z2).sum(-1)).sum()

def _keep_sum_impl(h_sg, delta, g, w, chunk=128):
    total = h_sg.new_zeros((), dtype=torch.float32)
    for s in range(0, h_sg.shape[0], chunk):
        sl = slice(s, s + chunk)
        total = total + _ck(_keep_chunk, h_sg[sl], delta[sl], g[sl], w)
    return total

@torch.no_grad()
def pack_topk(logits, k=256):
    lse = torch.logsumexp(logits.float(), -1)
    top = logits.float().topk(k, dim=-1)
    lp = top.values - lse[:, None]
    mass = (1.0 - lp.exp().sum(-1)).clamp(min=0.0)
    return (top.indices, lp, mass.clamp(min=1e-12).log())

@torch.no_grad()
def kl_to_packed(logits, ids, lp, tail):
    return _kl_packed(logits.float(), ids, lp, tail)

def ce_sum(*a, **k):
    with prof.rf('b:loss'):
        return _ce_sum_impl(*a, **k)

def corr_sum(*a, **k):
    with prof.rf('b:loss'):
        return _corr_sum_impl(*a, **k)

def keep_sum(*a, **k):
    with prof.rf('b:loss'):
        return _keep_sum_impl(*a, **k)