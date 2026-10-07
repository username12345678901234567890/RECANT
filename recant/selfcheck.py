from __future__ import annotations
import json
import sys

def check_quantize() -> dict:
    import torch
    from .model import fp4_kernels as k
    from .model import nvfp4 as q
    cases, ok = ([], True)
    torch.manual_seed(0)
    for dtype in (torch.bfloat16, torch.float32):
        for m, kk in ((300, 2048), (257, 6144), (17, 96)):
            for rot in (True, False):
                x = torch.randn(m, kk, device='cuda') * 3
                x[:, ::37] *= 30
                x = x.to(dtype).contiguous()
                ref = q.quantize(q.rht(x) if rot else x)
                got = k.quantize_fast(x, rot, False)
                same = torch.equal(ref.codes, got.codes) and torch.equal(ref.scales.view(torch.uint8), got.scales.view(torch.uint8)) and torch.equal(ref.gscale, got.gscale)
                cases.append({'dtype': str(dtype), 'shape': [m, kk], 'rht': rot, 'bit_identical': same})
                ok &= same
    x = (torch.randn(64, 256, device='cuda') * 2).to(torch.bfloat16).contiguous()
    target = q.rht(x).float()
    acc = sum((k.quantize_fast(x, True, True).dequant() for _ in range(64))) / 64
    bias = ((acc - target).abs().mean() / target.abs().mean()).item()
    sr_ok = bias < 0.03
    cases.append({'stochastic_mean_rel_err': bias, 'ok': sr_ok})
    return {'ok': bool(ok and sr_ok), 'cases': cases}
CHECKS = {'quantize': check_quantize}
if __name__ == '__main__':
    try:
        out = CHECKS[sys.argv[1]]()
    except Exception as e:
        out = {'ok': False, 'error': f'{type(e).__name__}: {str(e)[:300]}'}
    print('SELFCHECK_JSON ' + json.dumps(out))