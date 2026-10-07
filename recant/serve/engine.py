from __future__ import annotations
import json
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from ..branch import tap_indices

@dataclass
class Knobs:
    mode: str = 'learned'
    gate_fixed: float = 1.0
    gate_a: float | None = None
    gate_b: float | None = None
    delta_scale: float = 1.0
    gate_threshold: float | None = None

    def merged(self, over: dict | None) -> 'Knobs':
        if not over:
            return Knobs(**asdict(self))
        names = {f.name for f in fields(self)}
        bad = set(over) - names
        if bad:
            raise ValueError(f'unknown recant knob(s): {sorted(bad)}; valid: {sorted(names)}')
        k = Knobs(**{**asdict(self), **over})
        if k.mode not in ('learned', 'fixed', 'off'):
            raise ValueError('mode must be learned|fixed|off')
        return k

@dataclass
class GenParams:
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 0
    seed: int | None = None
    stop_ids: frozenset = frozenset()

@dataclass
class GenResult:
    tokens: list
    finish_reason: str
    prompt_tokens: int
    cached_tokens: int
    prefill_s: float
    decode_s: float
    gates: list | None = None
    mags: list | None = None

@dataclass
class _Snap:
    ids: np.ndarray
    cache: object
    rows: list

def sample_token(logits: torch.Tensor, temperature: float, top_p: float, top_k: int, gen) -> int:
    if temperature <= 0:
        return int(logits.argmax())
    l = logits.float() / temperature
    if top_k and top_k < l.numel():
        l = l.masked_fill(l < torch.topk(l, top_k).values[-1], float('-inf'))
    p = torch.softmax(l, -1)
    if top_p < 1.0:
        sp, idx = torch.sort(p, descending=True)
        sp = sp.masked_fill(torch.cumsum(sp, 0) - sp > top_p, 0.0)
        p = torch.zeros_like(p).scatter(0, idx, sp)
        p = p / p.sum()
    return int(torch.multinomial(p, 1, generator=gen))

class Engine:

    def __init__(self, model, heads, tap_idx, max_chunk: int=16384, max_ctx: int=98304, snap_capacity: int=6, knobs: Knobs | None=None, info: dict | None=None):
        if heads.n_taps != len(tap_idx) + 1:
            raise ValueError(f'heads expect {heads.n_taps} taps but the checkpoint lists {len(tap_idx)} tap layers + final')
        self.model, self.heads, self.tap_idx = (model, heads.eval(), list(tap_idx))
        self.max_chunk, self.max_ctx, self.snap_capacity = (max_chunk, max_ctx, snap_capacity)
        self.knobs = knobs or Knobs()
        self.info = info or {}
        self.device = model.embed_tokens.weight.device
        self._snaps: list[_Snap] = []
        self._tapset = set(self.tap_idx)

    @classmethod
    def load(cls, model_path, checkpoint_dir, device='cuda', dtype=torch.bfloat16, max_chunk=16384, max_ctx=98304, snap_capacity=6, log=print) -> 'Engine':
        from .. import backends, fast
        from ..ckpt import load_final
        from ..heads import Heads
        from ..model.config import TextConfig
        from ..model.qwen35 import load_hf
        ckpt = Path(checkpoint_dir)
        cfg = json.loads((ckpt / 'config.json').read_text())
        cfg_m = TextConfig.from_hf(model_path)
        rep = backends.select_all(device, cfg_m, log)
        rep['fast_paths'] = backends.select_fast_paths(device, cfg_m, log)
        model = load_hf(model_path, device=device, dtype=dtype, quantize=bool(cfg.get('fp4_backbone', True)), cfg=cfg_m, log=log)
        model.add_lora(cfg['lora']['r'], cfg['lora']['alpha'])
        h = cfg['heads']
        heads = Heads(cfg_m.hidden_size, n_taps=h['n_taps'], hidden=h['hidden'], n_bins=h['n_bins'], y_max=h['y_max']).to(device)
        load_final(ckpt, model, heads)
        tap_idx = cfg.get('tap_indices') or tap_indices(len(model.layers), (0.5, 0.7, 0.85))
        info = {'backends': {k: v.get('chosen', v) if isinstance(v, dict) else v for k, v in rep.items() if k != 'fast_paths'}, 'fast': fast.state(), 'checkpoint': str(ckpt), 'trained_steps': None}
        return cls(model, heads, tap_idx, max_chunk, max_ctx, snap_capacity, info=info)

    @torch.no_grad()
    def _forward(self, ids: torch.Tensor, cache, pos: int, rows: str='last') -> list:
        m = self.model
        sel = slice(-1, None) if rows == 'last' else slice(None)
        x = m.embed(ids[None])
        taps = {}
        for i, layer in enumerate(m.layers):
            if i in self._tapset:
                taps[i] = x[0, sel].clone()
            x = layer(x, cache.layers[i], pos)
        n = len(m.layers)
        if n in self._tapset:
            taps[n] = x[0, sel].clone()
        return [taps[i] for i in self.tap_idx] + [x[0, sel].clone()]

    @torch.no_grad()
    def _parts(self, rows: list, need_delta: bool=True):
        m, hd = (self.model, self.heads)
        h = m.final_norm(rows[-1])
        h_mix = hd.mix(rows)
        E = hd.expected_mag(hd.mag_logits(h_mix))
        z1 = m.logits(h).float()
        z2 = F.linear(hd.delta(h_mix).to(m.lm_head.weight.dtype), m.lm_head.weight).float() if need_delta else None
        return (z1, z2, E)

    def _gate(self, E: torch.Tensor, k: Knobs) -> torch.Tensor:
        if k.mode == 'off':
            return torch.zeros_like(E)
        if k.mode == 'fixed':
            g = torch.full_like(E, float(k.gate_fixed))
        else:
            a = float(self.heads.gate_a) if k.gate_a is None else k.gate_a
            b = float(self.heads.gate_b) if k.gate_b is None else k.gate_b
            g = torch.sigmoid(a * E + b)
        if k.gate_threshold is not None:
            g = torch.where(E >= k.gate_threshold, g, torch.zeros_like(g))
        return g * k.delta_scale

    def _combine(self, z1, z2, E, k: Knobs):
        g = self._gate(E, k)
        return (z1 if z2 is None else z1 + g[:, None] * z2, g)

    @torch.no_grad()
    def logits_last(self, rows: list, k: Knobs):
        z1, z2, E = self._parts(rows, need_delta=k.mode != 'off')
        lg, g = self._combine(z1, z2, E, k)
        return (lg[0], float(g[0]), float(E[0]))

    def _find_snap(self, ids: np.ndarray):
        best = None
        for s in self._snaps:
            n = len(s.ids)
            if n <= len(ids) and (best is None or n > len(best.ids)) and np.array_equal(ids[:n], s.ids):
                best = s
        return best

    def _store(self, snap: _Snap):
        if self.snap_capacity <= 0:
            return
        self._snaps = [s for s in self._snaps if not (len(s.ids) == len(snap.ids) and np.array_equal(s.ids, snap.ids))]
        self._snaps.append(snap)
        del self._snaps[:-self.snap_capacity]

    def prefill(self, prompt: list[int]):
        if not prompt:
            raise ValueError('empty prompt')
        ids = np.asarray(prompt, dtype=np.int64)
        snap = self._find_snap(ids)
        start = 0 if snap is None else len(snap.ids)
        cache = self.model.new_cache() if snap is None else snap.cache.fork()
        if snap is not None and start == len(ids):
            return (cache, snap.rows, start)
        t = torch.as_tensor(ids, device=self.device)
        rows = None
        for s in range(start, len(ids), self.max_chunk):
            rows = self._forward(t[s:s + self.max_chunk], cache, s)
        self._store(_Snap(ids.copy(), cache.fork(), rows))
        return (cache, rows, start)

    @torch.no_grad()
    def generate(self, prompt: list[int], p: GenParams, knobs: Knobs | None=None, on_token=None, record: bool=False) -> GenResult:
        k = knobs or self.knobs
        if len(prompt) >= self.max_ctx:
            raise ValueError(f'prompt has {len(prompt)} tokens; max_ctx is {self.max_ctx}')
        max_new = max(1, min(p.max_tokens, self.max_ctx - len(prompt)))
        gen = None
        if p.temperature > 0:
            gen = torch.Generator(device=self.device)
            gen.manual_seed(int(p.seed) if p.seed is not None else int(time.time_ns() % (1 << 62)))
        t0 = time.perf_counter()
        cache, rows, cached = self.prefill(prompt)
        t1 = time.perf_counter()
        fed = list(prompt)
        toks, gates, mags, reason = ([], [], [], 'length')
        for step in range(max_new):
            lg, g, e = self.logits_last(rows, k)
            tid = sample_token(lg, p.temperature, p.top_p, p.top_k, gen)
            if record:
                gates.append(g)
                mags.append(e)
            if tid in p.stop_ids:
                reason = 'stop'
                break
            toks.append(tid)
            if on_token is not None and on_token(tid, g, e):
                reason = 'stop'
                break
            if step == max_new - 1:
                break
            rows = self._forward(torch.tensor([tid], device=self.device), cache, len(fed))
            fed.append(tid)
        self._store(_Snap(np.asarray(fed, dtype=np.int64), cache, rows))
        return GenResult(toks, reason, len(prompt), cached, t1 - t0, time.perf_counter() - t1, gates if record else None, mags if record else None)

    @torch.no_grad()
    def score(self, prompt: list[int], completion: list[int], sweep: list[Knobs], row_chunk: int=256) -> dict:
        if not completion:
            raise ValueError('empty completion')
        cache, last, cached = self.prefill(prompt)
        rows = last
        if len(completion) > 1:
            t = torch.as_tensor(completion[:-1], dtype=torch.long, device=self.device)
            parts = []
            for s in range(0, len(t), self.max_chunk):
                parts.append(self._forward(t[s:s + self.max_chunk], cache, len(prompt) + s, rows='all'))
            rows = [torch.cat([last[j]] + [p[j] for p in parts], 0) for j in range(len(last))]
        tgt = torch.as_tensor(completion, dtype=torch.long, device=self.device)
        cfgs = [Knobs(mode='off')] + list(sweep)
        acc = [{'lp': 0.0, 'kl': 0.0, 'g': 0.0} for _ in cfgs]
        n = len(completion)
        for s in range(0, n, row_chunk):
            r = [x[s:s + row_chunk] for x in rows]
            z1, z2, E = self._parts(r, need_delta=any((c.mode != 'off' for c in cfgs)))
            base_lp = torch.log_softmax(z1, -1)
            y = tgt[s:s + row_chunk]
            for a, c in zip(acc, cfgs):
                lg, g = self._combine(z1, z2, E, c)
                lp = torch.log_softmax(lg, -1)
                a['lp'] += float(lp.gather(1, y[:, None]).sum())
                a['kl'] += float((base_lp.exp() * (base_lp - lp)).sum())
                a['g'] += float(g.sum())
        out = [{'knobs': asdict(c), 'logprob_sum': a['lp'], 'mean_logprob': a['lp'] / n, 'kl_to_base_mean': a['kl'] / n, 'gate_mean': a['g'] / n} for a, c in zip(acc, cfgs)]
        return {'n_prompt': len(prompt), 'n_completion': n, 'cached_tokens': cached, 'base': out[0], 'results': out[1:]}

    def describe(self) -> dict:
        h = self.heads
        return {'knobs': asdict(self.knobs), 'tap_layers': self.tap_idx, 'max_ctx': self.max_ctx, 'checkpoint_gate': {'a': float(h.gate_a), 'b': float(h.gate_b)}, 'y_max': h.y_max, 'snapshots': len(self._snaps), **self.info}