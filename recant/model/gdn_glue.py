from __future__ import annotations
import torch
import torch.nn.functional as F
_layout: str | None = None
_norm_ok = False

def set_conv_layout(layout: str | None) -> None:
    global _layout
    _layout = layout

def conv_ok() -> bool:
    return _layout is not None

def set_norm_ok(ok: bool) -> None:
    global _norm_ok
    _norm_ok = bool(ok)

def norm_ok() -> bool:
    return _norm_ok

def _to_fla(state, layout):
    return F.pad(state, (1, 0) if layout == 'zero_first' else (0, 1)).contiguous()

def _from_fla(fs, layout):
    return fs[..., 1:] if layout == 'zero_first' else fs[..., :-1]

def fla_conv(x_btc: torch.Tensor, weight_c1w: torch.Tensor, state, want_state: bool, layout: str | None=None):
    from fla.modules.conv import causal_conv1d
    layout = layout or _layout
    init = None if state is None else _to_fla(state.to(x_btc.dtype), layout)
    y, fs = causal_conv1d(x_btc.contiguous(), weight_c1w.squeeze(1).to(x_btc.dtype), None, initial_state=init, output_final_state=want_state, activation='silu')
    return (y, None if fs is None else _from_fla(fs, layout).contiguous())

def fla_gated_norm(x2d: torch.Tensor, gate2d: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    from fla.modules.fused_norm_gate import rms_norm_gated
    return rms_norm_gated(x2d, gate2d, w, None, 'swish', eps=eps)