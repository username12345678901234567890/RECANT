from __future__ import annotations
import torch
from .model import nvfp4, ops

def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12)).item()

def _try(fn):
    try:
        return (True, fn())
    except Exception as e:
        return (False, f'{type(e).__name__}: {str(e)[:300]}')

def select_fp4_gemm(device: str, log=print) -> dict:
    rep = {'chosen': 'emulated', 'candidates': {'cutlass_dsl': 'not implemented', 'transformer_engine': 'not implemented'}}
    if torch.device(device).type != 'cuda':
        rep['candidates']['scaled_mm'] = 'skipped (no CUDA device)'
        nvfp4.set_gemm_backend('emulated')
        return rep

    def check():
        torch.manual_seed(0)
        x = torch.randn(300, 1024, device=device, dtype=torch.bfloat16)
        w = torch.randn(2048, 1024, device=device, dtype=torch.bfloat16) * 0.05
        W = nvfp4.quantize_weight(w)
        a = nvfp4.quantize(nvfp4.rht(x))
        ref = nvfp4._gemm_emulated(a, W.fprop)
        got = nvfp4._gemm_scaled_mm(a, W.fprop)
        e1 = _rel(got, ref)
        dy = torch.randn(300, 2048, device=device, dtype=torch.bfloat16)
        d = nvfp4.quantize(nvfp4.rht(dy), stochastic=True)
        e2 = _rel(nvfp4._gemm_scaled_mm(d, W.dgrad), nvfp4._gemm_emulated(d, W.dgrad))
        if not (e1 < 0.03 and e2 < 0.03):
            raise ValueError(f'scaled_mm disagrees with emulated FP4: fprop {e1:.3g}, dgrad {e2:.3g}')
        return {'fprop_rel_err': e1, 'dgrad_rel_err': e2}
    ok, info = _try(check)
    rep['candidates']['scaled_mm'] = info if ok else f'FAILED: {info}'
    if ok:
        rep['chosen'] = 'scaled_mm'
    nvfp4.set_gemm_backend(rep['chosen'])
    log(f"[backends] FP4 GEMM -> {rep['chosen']} ({rep['candidates']['scaled_mm']})")
    return rep

def select_gdn(device: str, cfg, log=print) -> dict:
    rep = {'chosen': 'torch', 'candidates': {'flashqla': 'not implemented'}}
    if torch.device(device).type != 'cuda':
        rep['candidates']['fla'] = 'skipped (no CUDA device)'
        ops.set_gdn_backend('torch')
        return rep

    def check():
        torch.manual_seed(0)
        b, t, h = (1, 512, cfg.linear_num_value_heads)
        dk, dv = (cfg.linear_key_head_dim, cfg.linear_value_head_dim)
        mk = lambda *s: torch.randn(*s, device=device, dtype=torch.bfloat16, requires_grad=True)
        q, k, v = (mk(b, t, h, dk), mk(b, t, h, dk), mk(b, t, h, dv))
        g = -torch.rand(b, t, h, device=device) * 0.1
        beta = torch.rand(b, t, h, device=device, dtype=torch.bfloat16)
        o_ref, _ = ops.gdn_torch(q.float(), k.float(), v.float(), g, beta.float())
        o, _ = ops.gdn_fla(q, k, v, g, beta)
        e = _rel(o, o_ref)
        o.float().sum().backward()
        if not (e < 0.05 and torch.isfinite(q.grad).all()):
            raise ValueError(f'fla disagrees with torch reference (rel err {e:.3g})')
        return {'fwd_rel_err': e}
    ok, info = _try(check)
    rep['candidates']['fla'] = info if ok else f'FAILED: {info}'
    if ok:
        rep['chosen'] = 'fla'
    ops.set_gdn_backend(rep['chosen'])
    log(f"[backends] GDN -> {rep['chosen']} ({rep['candidates']['fla']})")
    return rep

def select_attention(device: str, cfg, log=print) -> dict:
    rep = {'chosen': 'sdpa_gqa', 'candidates': {'flash_attn_pkg': 'not used'}}
    if torch.device(device).type != 'cuda':
        rep['candidates']['sdpa'] = 'skipped (no CUDA device)'
        return rep

    def run(gqa):
        ops.set_attn_gqa(gqa)
        d, hq, hkv = (cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads)
        t = 8192
        q = torch.randn(1, hq, t, d, device=device, dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(1, hkv, t, d, device=device, dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(1, hkv, t, d, device=device, dtype=torch.bfloat16, requires_grad=True)
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        o = ops.attn_full(q, k, v)
        o.float().sum().backward()
        extra = (torch.cuda.max_memory_allocated() - base) / 2 ** 30
        if extra > 1.5:
            raise MemoryError(f'SDPA appears to use the O(T^2) math kernel (+{extra:.1f} GB at T={t})')
        if not (torch.isfinite(o).all() and torch.isfinite(q.grad).all() and torch.isfinite(k.grad).all()):
            raise ValueError('non-finite attention output/grad')
        c = ops.attn_cached(q[:, :, -256:].detach(), k.detach(), v.detach())
        if not torch.isfinite(c).all():
            raise ValueError('non-finite cached attention')
        return True
    for name, gqa in (('sdpa_gqa', True), ('sdpa_repeat_kv', False)):
        ok, info = _try(lambda gqa=gqa: run(gqa))
        rep['candidates'][name] = 'ok' if ok else f'FAILED: {info}'
        if ok:
            rep['chosen'] = name
            break
    else:
        raise RuntimeError(f"no working attention backend: {rep['candidates']}")
    log(f"[backends] attention -> {rep['chosen']}")
    return rep

def select_all(device: str, cfg, log=print) -> dict:
    return {'fp4_gemm': select_fp4_gemm(device, log), 'gdn': select_gdn(device, cfg, log), 'attention': select_attention(device, cfg, log)}

def _run_selfcheck(name: str, timeout: int=600) -> dict:
    import json
    import os
    import subprocess
    import sys
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join((p for p in sys.path if p))}
    try:
        r = subprocess.run([sys.executable, '-m', 'recant.selfcheck', name], capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {'ok': False, 'error': f'self-check timed out after {timeout}s'}
    for line in reversed(r.stdout.splitlines()):
        if line.startswith('SELFCHECK_JSON '):
            return json.loads(line[len('SELFCHECK_JSON '):])
    return {'ok': False, 'error': f'self-check process failed (rc={r.returncode}): {(r.stderr or r.stdout)[-300:]}'}

def select_quantizer(device: str, log=print) -> dict:
    from . import fast
    if not fast._requested['quantize']:
        return {'on': False, 'why': 'disabled by flag'}
    if torch.device(device).type != 'cuda':
        nvfp4.set_fast_quantizer(False)
        return {'on': False, 'why': 'no CUDA device (reference quantizer)'}
    res = _run_selfcheck('quantize')
    fast.set_verified('quantize', res.get('ok', False))
    nvfp4.set_fast_quantizer(fast.enabled('quantize'))
    log(f"[backends] fused NVFP4 quantizer -> {('ON' if fast.enabled('quantize') else 'off')} ({('bit-identical to reference' if res.get('ok') else res.get('error') or 'mismatch, see backends.json')})")
    return {'on': fast.enabled('quantize'), **res}

def select_elementwise(device: str, log=print) -> dict:
    from . import fast
    from .model import fused_ops as fo
    if not fast._requested['elementwise']:
        return {'on': False, 'why': 'disabled by flag'}
    if torch.device(device).type != 'cuda':
        return {'on': False, 'why': 'no CUDA device (eager reference)'}

    def check():
        fo.enable_compiled()
        torch.manual_seed(0)
        bf = dict(device=device, dtype=torch.bfloat16)
        cases = {'rmsnorm': ((torch.randn(777, 2048, **bf), torch.randn(2048, **bf) * 0.1, 1e-06), 1), 'rmsnorm_gated': ((torch.randn(1000, 128, **bf), torch.randn(1000, 128, **bf), torch.randn(128, **bf), 1e-06), 2), 'swiglu': ((torch.randn(777, 6144, **bf), torch.randn(777, 6144, **bf)), 2), 'gate_mul': ((torch.randn(777, 2048, **bf), torch.randn(777, 2048, **bf)), 2), 'gdn_gates': ((torch.randn(777, 16, **bf), torch.randn(16, device=device), torch.randn(16, device=device)), 1)}
        rep = {}
        for name, (args, _) in cases.items():
            leaf = [a.clone().requires_grad_(True) if isinstance(a, torch.Tensor) and a.is_floating_point() else a for a in args]
            leaf2 = [a.detach().clone().requires_grad_(True) if isinstance(a, torch.Tensor) and a.is_floating_point() else a for a in args]
            ref = fo.EAGER[name](*leaf)
            got = fo._impl[name](*leaf2)
            err = ((got.float() - ref.float()).norm() / ref.float().norm().clamp(min=1e-12)).item()
            ref.float().sum().backward()
            got.float().sum().backward()
            gerr = max((((b.grad.float() - a.grad.float()).norm() / a.grad.float().norm().clamp(min=1e-12)).item() for a, b in zip(leaf, leaf2) if isinstance(a, torch.Tensor) and a.grad is not None))
            rep[name] = {'fwd_rel_err': err, 'grad_rel_err': gerr}
            if not (err < 0.02 and gerr < 0.05):
                raise ValueError(f'{name}: compiled disagrees with eager ({err:.3g}, grad {gerr:.3g})')
        return rep
    ok, info = _try(check)
    fast.set_verified('elementwise', ok)
    if not ok or not fast.enabled('elementwise'):
        fo.disable_compiled()
    log(f"[backends] compiled elementwise -> {('ON' if ok else 'off')} ({('ok' if ok else info)})")
    return {'on': ok, 'detail': info}

def select_gdn_glue(device: str, cfg, log=print) -> dict:
    from . import fast
    from .model import gdn_glue
    if not fast._requested['gdn_glue']:
        return {'on': False, 'why': 'disabled by flag'}
    if torch.device(device).type != 'cuda' or ops.get_gdn_backend() != 'fla':
        return {'on': False, 'why': 'needs CUDA and the fla GDN backend'}
    rep: dict = {}

    def conv_check():
        torch.manual_seed(0)
        c, w = (cfg.conv_dim, cfg.linear_conv_kernel_dim)
        wt = torch.randn(c, 1, w, device=device, dtype=torch.bfloat16) * 0.3
        x = torch.randn(1, 300, c, device=device, dtype=torch.bfloat16)

        def ref(xx, prev):
            cat = torch.cat([prev, xx.transpose(1, 2)], -1)
            return (torch.nn.functional.silu(torch.nn.functional.conv1d(cat, wt, groups=c)).transpose(1, 2), cat[..., -(w - 1):])
        prev0 = torch.zeros(1, c, w - 1, device=device, dtype=torch.bfloat16)
        y1, s1 = ref(x[:, :180], prev0)
        y2, _ = ref(x[:, 180:], s1)
        for layout in ('zero_first', 'zero_last'):
            f1, fs1 = gdn_glue.fla_conv(x[:, :180], wt, None, True, layout)
            f2, _ = gdn_glue.fla_conv(x[:, 180:], wt, fs1, False, layout)
            e = max(((f1 - y1).float().norm() / y1.float().norm()).item(), ((f2 - y2).float().norm() / y2.float().norm()).item())
            rep[f'conv_{layout}_err'] = e
            if e < 0.02 and torch.equal(fs1.float(), s1.float()) or (e < 0.02 and (fs1.float() - s1.float()).abs().max() < 0.01):
                return layout
        raise ValueError(f'no fla conv state layout matches the reference: {rep}')
    ok, layout = _try(conv_check)
    gdn_glue.set_conv_layout(layout if ok else None)
    rep['conv'] = layout if ok else f'FAILED: {layout}'

    def norm_check():
        from .model.layers import RMSNormGated
        d = cfg.linear_value_head_dim
        m = RMSNormGated(d, cfg.rms_norm_eps).to(device=device, dtype=torch.bfloat16)
        m.weight.data.normal_(1.0, 0.1)
        x = torch.randn(512, d, device=device, dtype=torch.bfloat16)
        g = torch.randn(512, d, device=device, dtype=torch.bfloat16)
        ref = m(x, g)
        got = gdn_glue.fla_gated_norm(x, g, m.weight, cfg.rms_norm_eps)
        e = _rel(got, ref)
        if e > 0.02:
            raise ValueError(f'fla gated norm disagrees with reference ({e:.3g})')
        return e
    ok2, e = _try(norm_check)
    gdn_glue.set_norm_ok(ok2)
    rep['norm'] = e if ok2 else f'FAILED: {e}'
    fast.set_verified('gdn_glue', ok or ok2)
    log(f"[backends] fla GDN glue -> conv {('ON' if ok else 'off')}, gated norm {('ON' if ok2 else 'off')}")
    return {'on': ok or ok2, **rep}

def select_attention_fast(device: str, cfg, log=print) -> dict:
    from . import fast
    if not fast._requested['attn']:
        return {'on': False, 'why': 'disabled by flag'}
    if torch.device(device).type != 'cuda':
        return {'on': False, 'why': 'no CUDA device'}

    def check():
        torch.manual_seed(0)
        hq, hkv, d = (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim)
        q = torch.randn(1, hq, 700, d, device=device, dtype=torch.bfloat16)
        k = torch.randn(1, hkv, 1900, d, device=device, dtype=torch.bfloat16)
        v = torch.randn(1, hkv, 1900, d, device=device, dtype=torch.bfloat16)
        ref = ops.attn_cached_masked(q, k, v)
        got = ops.attn_cached_two_part(q, k, v, ops._attn_lse_flash)
        e = _rel(got, ref)
        if e > 0.02:
            raise ValueError(f'flash+LSE cached attention disagrees with masked reference ({e:.3g})')
        return e
    ok, info = _try(check)
    fast.set_verified('attn', ok)
    ops.set_restrict_sdpa(fast.enabled('attn'))
    ops.set_cached_impl('flash_lse' if fast.enabled('attn') else 'masked')
    log(f"[backends] cached attention -> {('flash+LSE' if ok else 'masked SDPA')} ({('rel err %.2g' % info if ok else info)})")
    return {'on': ok, 'detail': info}

def select_fast_paths(device: str, cfg, log=print) -> dict:
    from . import fast
    rep = {'quantize': select_quantizer(device, log), 'elementwise': select_elementwise(device, log), 'gdn_glue': select_gdn_glue(device, cfg, log), 'attn': select_attention_fast(device, cfg, log), 'lora': {'on': fast.enabled('lora'), 'why': 'pure torch, equivalence unit-tested'}}
    rep['flags'] = fast.state()
    return rep