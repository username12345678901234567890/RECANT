from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from .. import prof
BLOCK = 16
FP4_MAX = 6.0
E4M3_MAX = 448.0
FP4_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_MIDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)

def _grid(device):
    return torch.tensor(FP4_GRID, device=device)

def _hadamard16() -> torch.Tensor:
    h = torch.ones(1, 1)
    for _ in range(4):
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / 4.0
_H16 = _hadamard16()
_SIGNS = (torch.randint(0, 2, (16,), generator=torch.Generator().manual_seed(1234)) * 2 - 1).float()

def rht(x: torch.Tensor) -> torch.Tensor:
    k = x.shape[-1]
    lead = (*x.shape[:-1], k // BLOCK)
    y = x.float().reshape(*lead, BLOCK) * _SIGNS.to(x.device)
    for s in (1, 2, 4, 8):
        y = y.reshape(*lead, BLOCK // (2 * s), 2, s)
        a, b = (y[..., 0, :], y[..., 1, :])
        y = torch.stack([a + b, a - b], dim=-2)
    return (y.reshape(*lead, BLOCK) * 0.25).reshape(x.shape).to(x.dtype)

def rht_matmul(x: torch.Tensor) -> torch.Tensor:
    k = x.shape[-1]
    y = x.float().reshape(*x.shape[:-1], k // BLOCK, BLOCK) * _SIGNS.to(x.device)
    return (y @ _H16.to(x.device)).reshape(x.shape).to(x.dtype)

@dataclass
class FP4Tensor:
    codes: torch.Tensor
    scales: torch.Tensor
    gscale: torch.Tensor
    shape: tuple

    def dequant(self, dtype=torch.float32, rows_per_chunk: int=1 << 26) -> torch.Tensor:
        m, k = self.shape
        step = max(1, rows_per_chunk // k)
        g = _grid(self.codes.device)
        out = torch.empty(m, k, dtype=dtype, device=self.codes.device)
        for s in range(0, m, step):
            codes = self.codes[s:s + step]
            lo, hi = (codes & 15, codes >> 4)
            c = torch.stack([lo, hi], -1).reshape(codes.shape[0], k)
            val = g[(c & 7).long()] * torch.where(c & 8 > 0, -1.0, 1.0)
            sc = self.scales[s:s + step].float().repeat_interleave(BLOCK, dim=-1) * self.gscale
            out[s:s + step] = (val * sc).to(dtype)
        return out

def _round_codes(a_scaled: torch.Tensor, stochastic: bool) -> torch.Tensor:
    neg = a_scaled < 0
    a = a_scaled.abs().clamp(max=FP4_MAX)
    if stochastic:
        grid = _grid(a.device)
        hi = torch.bucketize(a, grid).clamp(min=1)
        lo = hi - 1
        glo, ghi = (grid[lo], grid[hi])
        p = (a - glo) / (ghi - glo)
        idx = torch.where(torch.rand_like(a) < p, hi, lo)
    else:
        idx = torch.bucketize(a, torch.tensor(_MIDS, device=a.device))
    return (idx + 8 * neg.long()).to(torch.uint8)

def _pack(codes: torch.Tensor) -> torch.Tensor:
    return codes[..., 0::2] | codes[..., 1::2] << 4

def quantize(x: torch.Tensor, stochastic: bool=False, rows_per_chunk: int=1 << 26) -> FP4Tensor:
    assert x.dim() == 2 and x.shape[1] % BLOCK == 0, x.shape
    m, k = x.shape
    s_g = x.float().abs().max().clamp(min=1e-30) / (FP4_MAX * E4M3_MAX)
    step = max(1, rows_per_chunk // k)
    codes_out = torch.empty(m, k // 2, dtype=torch.uint8, device=x.device)
    scales_out = torch.empty(m, k // BLOCK, dtype=torch.float8_e4m3fn, device=x.device)
    for s in range(0, m, step):
        xb = x[s:s + step].float().view(-1, k // BLOCK, BLOCK)
        s_b = (xb.abs().amax(-1) / FP4_MAX / s_g).clamp(max=E4M3_MAX)
        s8 = s_b.to(torch.float8_e4m3fn)
        sq = s8.float()
        denom = torch.where(sq == 0, torch.ones_like(sq), sq) * s_g
        codes_out[s:s + step] = _pack(_round_codes(xb / denom[..., None], stochastic).view(xb.shape[0], k))
        scales_out[s:s + step] = s8
    return FP4Tensor(codes_out, scales_out, s_g.reshape(()).clone(), (m, k))

@dataclass
class FP4Weight:
    fprop: FP4Tensor
    dgrad: FP4Tensor
    out_features: int
    in_features: int
    use_rht: bool

def quantize_weight(w: torch.Tensor, use_rht: bool=True) -> FP4Weight:
    n, k = w.shape
    wf = w.float()
    fp = quantize(rht(wf) if use_rht else wf)
    dg = quantize(rht(wf.t().contiguous()) if use_rht else wf.t().contiguous())
    return FP4Weight(fp, dg, n, k, use_rht)

def _gemm_emulated(a: FP4Tensor, b: FP4Tensor) -> torch.Tensor:
    dt = torch.float32 if a.codes.device.type == 'cpu' else torch.bfloat16
    return (a.dequant(dt) @ b.dequant(dt).t()).float()

def _to_blocked(s: torch.Tensor) -> torch.Tensor:
    rows, cols = s.shape
    nrb, ncb = (-(-rows // 128), -(-cols // 4))
    u = s.view(torch.uint8)
    if rows % 128 or cols % 4:
        u = F.pad(u, (0, ncb * 4 - cols, 0, nrb * 128 - rows))
    blocks = u.view(nrb, 128, ncb, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten().view(torch.float8_e4m3fn)

def _gemm_scaled_mm(a: FP4Tensor, b: FP4Tensor) -> torch.Tensor:
    fp4 = torch.float4_e2m1fn_x2
    out = torch._scaled_mm(a.codes.view(fp4), b.codes.view(fp4).t(), scale_a=_to_blocked(a.scales), scale_b=_to_blocked(b.scales), out_dtype=torch.bfloat16)
    return out.float() * (a.gscale * b.gscale)
GEMM_BACKENDS = {'emulated': _gemm_emulated, 'scaled_mm': _gemm_scaled_mm}
_state = {'gemm': 'emulated'}

def set_gemm_backend(name: str) -> None:
    if name not in GEMM_BACKENDS:
        raise KeyError(f'unknown FP4 GEMM backend {name!r}; have {list(GEMM_BACKENDS)}')
    _state['gemm'] = name

def get_gemm_backend() -> str:
    return _state['gemm']

def gemm_fp4(a: FP4Tensor, b: FP4Tensor) -> torch.Tensor:
    with prof.rf('b:fp4_gemm'):
        return GEMM_BACKENDS[_state['gemm']](a, b)
_fast = {'quantize': False}

def set_fast_quantizer(flag: bool) -> None:
    _fast['quantize'] = bool(flag)

def quantize_activation(x: torch.Tensor, use_rht: bool=True, stochastic: bool=False) -> FP4Tensor:
    with prof.rf('b:quantize'):
        if _fast['quantize'] and x.is_cuda and (x.dim() == 2) and (x.shape[1] % BLOCK == 0):
            from . import fp4_kernels
            return fp4_kernels.quantize_fast(x.contiguous(), use_rht, stochastic)
        return quantize(rht(x) if use_rht else x, stochastic)

class FP4Linear(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, w: FP4Weight):
        ctx.w = w
        x2 = x.reshape(-1, w.in_features)
        a = quantize_activation(x2, w.use_rht)
        y = gemm_fp4(a, w.fprop)
        return y.reshape(*x.shape[:-1], w.out_features).to(x.dtype)

    @staticmethod
    def backward(ctx, dy):
        w = ctx.w
        d2 = dy.reshape(-1, w.out_features)
        d = quantize_activation(d2, w.use_rht, stochastic=True)
        dx = gemm_fp4(d, w.dgrad)
        return (dx.reshape(*dy.shape[:-1], w.in_features).to(dy.dtype), None)

def fp4_linear(x: torch.Tensor, w: FP4Weight) -> torch.Tensor:
    return FP4Linear.apply(x, w)