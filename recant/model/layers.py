from __future__ import annotations
import math
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import fast, prof
from . import fused_ops as fo
from . import gdn_glue, nvfp4
from .config import TextConfig
from .ops import attn_cached, attn_full, gdn_core

class RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-06):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim), requires_grad=False)

    def forward(self, x):
        return fo.rmsnorm(x, self.weight, self.eps)

class RMSNormGated(nn.Module):

    def __init__(self, dim, eps=1e-06):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim), requires_grad=False)

    def forward(self, x, gate):
        if fast.enabled('gdn_glue') and gdn_glue.norm_ok() and x.is_cuda:
            with prof.rf('b:elementwise'):
                return gdn_glue.fla_gated_norm(x, gate, self.weight, self.eps)
        return fo.rmsnorm_gated(x, gate, self.weight, self.eps)

class QLinear(nn.Module):

    def __init__(self, in_f, out_f, lora_ok=True):
        super().__init__()
        self.in_features, self.out_features, self.lora_ok = (in_f, out_f, lora_ok)
        self.weight = nn.Parameter(torch.empty(out_f, in_f), requires_grad=False)
        self.fp4 = None
        self.lora_A = self.lora_B = None
        self.lora_scale = 1.0

    @property
    def quantizable(self):
        return self.in_features % 16 == 0 and self.out_features % 16 == 0 and (self.out_features >= 256)

    def quantize_(self, use_rht=True):
        if self.fp4 is None and self.quantizable:
            self.fp4 = nvfp4.quantize_weight(self.weight.data, use_rht)
            self.weight = None

    def add_lora(self, r, alpha):
        dev = self.fp4.fprop.codes.device if self.fp4 is not None else self.weight.device
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, device=dev, dtype=torch.float32))
        self.lora_scale = alpha / r

    def base(self, x):
        return nvfp4.fp4_linear(x, self.fp4) if self.fp4 is not None else F.linear(x, self.weight)

    def forward(self, x):
        y = self.base(x)
        if self.lora_A is not None:
            with prof.rf('b:lora'):
                y = y + F.linear(F.linear(x, self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype)) * self.lora_scale
        return y

def grouped_linear(x, lins):
    ys = [l.base(x) for l in lins]
    if not any((l.lora_A is not None for l in lins)):
        return ys
    if not fast.enabled('lora'):
        outs = []
        for l, y in zip(lins, ys):
            if l.lora_A is not None:
                with prof.rf('b:lora'):
                    y = y + F.linear(F.linear(x, l.lora_A.to(x.dtype)), l.lora_B.to(x.dtype)) * l.lora_scale
            outs.append(y)
        return outs
    with prof.rf('b:lora'):
        act = [l for l in lins if l.lora_A is not None]
        h = F.linear(x, torch.cat([l.lora_A for l in act], 0).to(x.dtype))
        off, outs = (0, [])
        for l, y in zip(lins, ys):
            if l.lora_A is None:
                outs.append(y)
                continue
            r = l.lora_A.shape[0]
            hi = h[..., off:off + r].reshape(-1, r)
            off += r
            o = torch.addmm(y.reshape(-1, l.out_features), hi, l.lora_B.to(x.dtype).t(), alpha=l.lora_scale)
            outs.append(o.reshape(*y.shape))
        return outs

class MLP(nn.Module):

    def __init__(self, c: TextConfig):
        super().__init__()
        self.gate_proj = QLinear(c.hidden_size, c.intermediate_size)
        self.up_proj = QLinear(c.hidden_size, c.intermediate_size)
        self.down_proj = QLinear(c.intermediate_size, c.hidden_size)

    def forward(self, x):
        g, u = grouped_linear(x, [self.gate_proj, self.up_proj])
        return self.down_proj(fo.swiglu(g, u))

@dataclass
class LayerCache:
    conv_state: torch.Tensor | None = None
    rec_state: torch.Tensor | None = None
    k: torch.Tensor | None = None
    v: torch.Tensor | None = None

@dataclass
class SeqCache:
    layers: list = field(default_factory=list)

    @classmethod
    def new(cls, n_layers):
        return cls([LayerCache() for _ in range(n_layers)])

    def fork(self) -> 'SeqCache':
        out = []
        for c in self.layers:
            out.append(LayerCache(None if c.conv_state is None else c.conv_state.clone(), None if c.rec_state is None else c.rec_state.clone(), c.k, c.v))
        return SeqCache(out)

class GatedDeltaNet(nn.Module):

    def __init__(self, c: TextConfig):
        super().__init__()
        self.c = c
        self.in_proj_qkv = QLinear(c.hidden_size, c.conv_dim)
        self.in_proj_z = QLinear(c.hidden_size, c.value_dim)
        self.in_proj_b = QLinear(c.hidden_size, c.linear_num_value_heads, lora_ok=False)
        self.in_proj_a = QLinear(c.hidden_size, c.linear_num_value_heads, lora_ok=False)
        self.out_proj = QLinear(c.value_dim, c.hidden_size)
        self.conv1d = nn.Conv1d(c.conv_dim, c.conv_dim, c.linear_conv_kernel_dim, groups=c.conv_dim, bias=False, padding=0)
        self.conv1d.weight.requires_grad_(False)
        self.dt_bias = nn.Parameter(torch.ones(c.linear_num_value_heads), requires_grad=False)
        self.A_log = nn.Parameter(torch.zeros(c.linear_num_value_heads), requires_grad=False)
        self.norm = RMSNormGated(c.linear_value_head_dim, c.rms_norm_eps)

    def forward(self, x, cache: LayerCache | None=None, pos_offset: int=0):
        c = self.c
        b, t, _ = x.shape
        km1 = c.linear_conv_kernel_dim - 1
        qkv, z = grouped_linear(x, [self.in_proj_qkv, self.in_proj_z])
        z = z.reshape(b, t, -1, c.linear_value_head_dim)
        beta = self.in_proj_b(x).sigmoid()
        a = self.in_proj_a(x)
        prev = cache.conv_state if cache is not None else None
        if fast.enabled('gdn_glue') and gdn_glue.conv_ok() and qkv.is_cuda:
            with prof.rf('b:elementwise'):
                conv, new_state = gdn_glue.fla_conv(qkv, self.conv1d.weight, prev, cache is not None)
            if cache is not None:
                cache.conv_state = new_state.detach().clone()
        else:
            mixed = qkv.transpose(1, 2)
            prev = prev if prev is not None else torch.zeros(b, c.conv_dim, km1, dtype=mixed.dtype, device=mixed.device)
            cat = torch.cat([prev.to(mixed.dtype), mixed], dim=-1)
            conv = F.silu(F.conv1d(cat, self.conv1d.weight.to(mixed.dtype), groups=c.conv_dim)).transpose(1, 2)
            if cache is not None:
                cache.conv_state = cat[..., -km1:].detach().clone()
        q, k, v = torch.split(conv, [c.key_dim, c.key_dim, c.value_dim], dim=-1)
        q = q.reshape(b, t, -1, c.linear_key_head_dim)
        k = k.reshape(b, t, -1, c.linear_key_head_dim)
        v = v.reshape(b, t, -1, c.linear_value_head_dim)
        g = fo.gdn_gates(a, self.A_log, self.dt_bias)
        rep = c.linear_num_value_heads // c.linear_num_key_heads
        if rep > 1:
            q, k = (q.repeat_interleave(rep, dim=2), k.repeat_interleave(rep, dim=2))
        init = cache.rec_state if cache is not None else None
        core, state = gdn_core(q, k, v, g, beta, initial_state=init, output_final_state=cache is not None)
        if cache is not None:
            cache.rec_state = state.detach()
        core = self.norm(core.reshape(-1, c.linear_value_head_dim), z.reshape(-1, c.linear_value_head_dim))
        return self.out_proj(core.reshape(b, t, -1))

def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)

def rope_cos_sin(positions: torch.Tensor, rot_dim: int, theta: float):
    inv = 1.0 / theta ** (torch.arange(0, rot_dim, 2, dtype=torch.float32, device=positions.device) / rot_dim)
    f = positions.float()[:, None] * inv[None, :]
    emb = torch.cat([f, f], dim=-1)
    return (emb.cos(), emb.sin())

def apply_rope(x, cos, sin):
    return fo.rope(x, cos, sin)

class Attention(nn.Module):

    def __init__(self, c: TextConfig):
        super().__init__()
        self.c = c
        d = c.head_dim
        self.q_proj = QLinear(c.hidden_size, c.num_attention_heads * d * 2)
        self.k_proj = QLinear(c.hidden_size, c.num_key_value_heads * d)
        self.v_proj = QLinear(c.hidden_size, c.num_key_value_heads * d)
        self.o_proj = QLinear(c.num_attention_heads * d, c.hidden_size)
        self.q_norm = RMSNorm(d, c.rms_norm_eps)
        self.k_norm = RMSNorm(d, c.rms_norm_eps)

    def forward(self, x, cache: LayerCache | None=None, pos_offset: int=0):
        c = self.c
        b, t, _ = x.shape
        d = c.head_dim
        qg, kp, vp = grouped_linear(x, [self.q_proj, self.k_proj, self.v_proj])
        q, gate = qg.view(b, t, -1, d * 2).chunk(2, dim=-1)
        gate = gate.reshape(b, t, -1)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(kp.view(b, t, -1, d)).transpose(1, 2)
        v = vp.view(b, t, -1, d).transpose(1, 2)
        pos = torch.arange(pos_offset, pos_offset + t, device=x.device)
        cos, sin = rope_cos_sin(pos, c.rotary_dim, c.rope_theta)
        q, k = (apply_rope(q, cos, sin), apply_rope(k, cos, sin))
        if cache is not None:
            if cache.k is not None:
                k, v = (torch.cat([cache.k, k], dim=2), torch.cat([cache.v, v], dim=2))
            cache.k, cache.v = (k.detach(), v.detach())
            o = attn_cached(q, k, v)
        else:
            o = attn_full(q, k, v)
        o = fo.gate_mul(o.transpose(1, 2).reshape(b, t, -1), gate)
        return self.o_proj(o)

class DecoderLayer(nn.Module):

    def __init__(self, c: TextConfig, idx: int):
        super().__init__()
        self.idx = idx
        self.kind = c.layer_types[idx]
        if self.kind == 'linear_attention':
            self.linear_attn = GatedDeltaNet(c)
        else:
            self.self_attn = Attention(c)
        self.mlp = MLP(c)
        self.input_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)

    @property
    def mixer(self):
        return self.linear_attn if self.kind == 'linear_attention' else self.self_attn

    def forward(self, x, cache: LayerCache | None=None, pos_offset: int=0):
        x = x + self.mixer(self.input_layernorm(x), cache, pos_offset)
        return x + self.mlp(self.post_attention_layernorm(x))