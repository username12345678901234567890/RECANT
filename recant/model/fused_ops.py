"""Elementwise ops of the decoder layer: eager reference + optional torch.compile versions.

The eager functions are the reference (they reproduce the original module code exactly). `enable_compiled`
swaps in `torch.compile(dynamic=True)` versions after a startup self-check (backends.select_elementwise);
only pure elementwise functions are compiled, never autograd Functions, so there are no graph breaks.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .. import prof


def rmsnorm_eager(x, w, eps):
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
    return (y * (1.0 + w.float())).type_as(x)


def rmsnorm_gated_eager(x, gate, w, eps):
    dt = x.dtype
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
    y = w * y.to(dt)
    return (y * F.silu(gate.float())).to(dt)


def swiglu_eager(g, u):
    return F.silu(g) * u


def gate_mul_eager(o, gate):
    return o * torch.sigmoid(gate)


def rope_eager(x, cos, sin):
    """x [B,H,T,D]; rotates the first cos.shape[-1] dims. cos/sin [T, r]."""
    r = cos.shape[-1]
    xr, xp = x[..., :r], x[..., r:]
    cos, sin = cos.to(x.dtype)[None, None], sin.to(x.dtype)[None, None]
    h = r // 2
    rot = torch.cat((-xr[..., h:], xr[..., :h]), dim=-1)
    return torch.cat([xr * cos + rot * sin, xp], dim=-1)


def gdn_gates_eager(a, A_log, dt_bias):
    return -A_log.float().exp() * F.softplus(a.float() + dt_bias.float())


EAGER = {"rmsnorm": rmsnorm_eager, "rmsnorm_gated": rmsnorm_gated_eager, "swiglu": swiglu_eager,
         "gate_mul": gate_mul_eager, "rope": rope_eager, "gdn_gates": gdn_gates_eager}
_impl = dict(EAGER)
_compiled = False


def enable_compiled() -> None:
    global _compiled
    for name, fn in EAGER.items():
        _impl[name] = torch.compile(fn, dynamic=True)
    _compiled = True


def disable_compiled() -> None:
    global _compiled
    _impl.update(EAGER)
    _compiled = False


def is_compiled() -> bool:
    return _compiled


def rmsnorm(x, w, eps):
    with prof.rf("b:elementwise"):
        return _impl["rmsnorm"](x, w, eps)


def rmsnorm_gated(x, gate, w, eps):
    with prof.rf("b:elementwise"):
        return _impl["rmsnorm_gated"](x, gate, w, eps)


def swiglu(g, u):
    with prof.rf("b:elementwise"):
        return _impl["swiglu"](g, u)


def gate_mul(o, gate):
    with prof.rf("b:elementwise"):
        return _impl["gate_mul"](o, gate)


def rope(x, cos, sin):
    with prof.rf("b:elementwise"):
        return _impl["rope"](x, cos, sin)


def gdn_gates(a, A_log, dt_bias):
    with prof.rf("b:elementwise"):
        return _impl["gdn_gates"](a, A_log, dt_bias)
