"""Startup backend selection (no benchmarking): take the FIRST candidate that reproduces a reference.

FP4 GEMM: scaled_mm (cuBLASLt) -> emulated.   GDN: fla -> torch.   Attention: SDPA (GQA) -> SDPA (repeat_kv).
`cutlass_dsl` and Transformer Engine paths are NOT implemented; they are listed as unavailable in the report
so it is explicit that the fallback chain is shorter than the design doc's.
Everything runs inside try/except; failures and reasons go to backends.json.
"""
from __future__ import annotations

import traceback

import torch

from .model import nvfp4, ops


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12)).item()


def _try(fn):
    try:
        return True, fn()
    except Exception as e:  # noqa: BLE001 - any failure just disqualifies the candidate
        return False, f"{type(e).__name__}: {str(e)[:300]}"


def select_fp4_gemm(device: str, log=print) -> dict:
    rep = {"chosen": "emulated", "candidates": {"cutlass_dsl": "not implemented", "transformer_engine": "not implemented"}}
    if torch.device(device).type != "cuda":
        rep["candidates"]["scaled_mm"] = "skipped (no CUDA device)"
        nvfp4.set_gemm_backend("emulated")
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
        if not (e1 < 3e-2 and e2 < 3e-2):
            raise ValueError(f"scaled_mm disagrees with emulated FP4: fprop {e1:.3g}, dgrad {e2:.3g}")
        return {"fprop_rel_err": e1, "dgrad_rel_err": e2}

    ok, info = _try(check)
    rep["candidates"]["scaled_mm"] = info if ok else f"FAILED: {info}"
    if ok:
        rep["chosen"] = "scaled_mm"
    nvfp4.set_gemm_backend(rep["chosen"])
    log(f"[backends] FP4 GEMM -> {rep['chosen']} ({rep['candidates']['scaled_mm']})")
    return rep


def select_gdn(device: str, cfg, log=print) -> dict:
    rep = {"chosen": "torch", "candidates": {"flashqla": "not implemented"}}
    if torch.device(device).type != "cuda":
        rep["candidates"]["fla"] = "skipped (no CUDA device)"
        ops.set_gdn_backend("torch")
        return rep

    def check():
        torch.manual_seed(0)
        b, t, h = 1, 512, cfg.linear_num_value_heads
        dk, dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
        mk = lambda *s: torch.randn(*s, device=device, dtype=torch.bfloat16, requires_grad=True)
        q, k, v = mk(b, t, h, dk), mk(b, t, h, dk), mk(b, t, h, dv)
        g = -torch.rand(b, t, h, device=device) * 0.1
        beta = torch.rand(b, t, h, device=device, dtype=torch.bfloat16)
        o_ref, _ = ops.gdn_torch(q.float(), k.float(), v.float(), g, beta.float())
        o, _ = ops.gdn_fla(q, k, v, g, beta)
        e = _rel(o, o_ref)
        o.float().sum().backward()                                   # backward must work too
        if not (e < 5e-2 and torch.isfinite(q.grad).all()):
            raise ValueError(f"fla disagrees with torch reference (rel err {e:.3g})")
        return {"fwd_rel_err": e}

    ok, info = _try(check)
    rep["candidates"]["fla"] = info if ok else f"FAILED: {info}"
    if ok:
        rep["chosen"] = "fla"
    ops.set_gdn_backend(rep["chosen"])
    log(f"[backends] GDN -> {rep['chosen']} ({rep['candidates']['fla']})")
    return rep


def select_attention(device: str, cfg, log=print) -> dict:
    rep = {"chosen": "sdpa_gqa", "candidates": {"flash_attn_pkg": "not used"}}
    if torch.device(device).type != "cuda":
        rep["candidates"]["sdpa"] = "skipped (no CUDA device)"
        return rep

    def run(gqa):
        ops.set_attn_gqa(gqa)
        d, hq, hkv = cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads
        # 8192 tokens: a flash/efficient kernel needs O(T) extra memory, the math fallback materializes
        # hq*T*T scores (>= 2 GB in bf16) which would be fatal at 98K tokens - detect it here, not at step 1.
        t = 8192
        q = torch.randn(1, hq, t, d, device=device, dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(1, hkv, t, d, device=device, dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(1, hkv, t, d, device=device, dtype=torch.bfloat16, requires_grad=True)
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        o = ops.attn_full(q, k, v)
        o.float().sum().backward()
        extra = (torch.cuda.max_memory_allocated() - base) / 2**30
        if extra > 1.5:
            raise MemoryError(f"SDPA appears to use the O(T^2) math kernel (+{extra:.1f} GB at T={t})")
        if not (torch.isfinite(o).all() and torch.isfinite(q.grad).all() and torch.isfinite(k.grad).all()):
            raise ValueError("non-finite attention output/grad")
        c = ops.attn_cached(q[:, :, -256:].detach(), k.detach(), v.detach())    # offset-causal path
        if not torch.isfinite(c).all():
            raise ValueError("non-finite cached attention")
        return True

    for name, gqa in (("sdpa_gqa", True), ("sdpa_repeat_kv", False)):
        ok, info = _try(lambda gqa=gqa: run(gqa))
        rep["candidates"][name] = "ok" if ok else f"FAILED: {info}"
        if ok:
            rep["chosen"] = name
            break
    else:
        raise RuntimeError(f"no working attention backend: {rep['candidates']}")
    log(f"[backends] attention -> {rep['chosen']}")
    return rep


def select_all(device: str, cfg, log=print) -> dict:
    return {"fp4_gemm": select_fp4_gemm(device, log), "gdn": select_gdn(device, cfg, log),
            "attention": select_attention(device, cfg, log)}
