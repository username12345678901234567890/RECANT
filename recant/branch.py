"""One trajectory: unified prefill pass (main stream + answer-hinted branches), losses, layerwise backward.

Pass A (no grad, chunked with a cache) does three jobs at once:
  1. runs the main stream and RECORDS every layer's input (these are the activation checkpoints),
  2. at each sampled commit turn forks the cache (GDN state snapshot + shared attention KV),
     runs `hint + turn` on the fork and keeps only the top-k `p_full` over the commit span,
  3. leaves the final hidden states from which p_plain (and the magnitude target) are computed.
The branch never touches the main cache, so the main stream is exactly the un-hinted model
(design invariant 1: targets come from a pass with delta off).

Then the heads/losses run on gathered rows, and a manual reverse loop recomputes each layer with
grad from its recorded input (2 forwards + 1 backward in total, no separate branch forward).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from .data.render import (F_COMMIT, T_ANCHOR, T_CALL_END, T_CALL_START, T_CE_END, T_CE_START, T_FLAGS,
                          T_START)
from . import prof
from .heads import GradScale
from .losses import ce_sum, corr_sum, keep_sum, kl_to_packed, pack_topk


@dataclass
class StepCfg:
    lambda_corr: float = 1.0
    lambda_keep: float = 0.1
    lambda_mag: float = 0.5
    max_branch_turns: int = 16
    n_keep: int = 256
    topk: int = 256
    max_chunk: int = 16384
    tap_fracs: tuple = (0.5, 0.7, 0.85)
    keep_assistant_frac: float = 0.9


@dataclass
class Plan:
    n: int
    ce_pos: np.ndarray
    ce_tgt: np.ndarray
    spans: list                 # (call_start, call_end, turn_start, call_end) per sampled turn
    corr_pos: np.ndarray        # concat over spans of [cs-1, ce-1)
    anchor_pos: np.ndarray      # cs-1 per span
    keep_pos: np.ndarray
    P: np.ndarray = None        # sorted union of every position that needs a final hidden row
    H: np.ndarray = None        # sorted union of head positions (corr + keep)

    @property
    def counts(self) -> dict:
        return {"ce": len(self.ce_pos), "corr": len(self.corr_pos), "keep": len(self.keep_pos),
                "anchor": len(self.anchor_pos)}


def tap_indices(n_layers: int, fracs) -> list[int]:
    """Indices into the recorded activations (Xs[i] = output of layer i-1) used as taps."""
    return [min(n_layers, max(1, round(f * n_layers))) for f in fracs]


def plan_trajectory(traj: dict, cfg: StepCfg, rng: np.random.Generator) -> Plan:
    turns = np.asarray(traj["turns"])
    ids = np.asarray(traj["ids"])
    n = len(ids)
    commit = [k for k in range(len(turns)) if turns[k, T_FLAGS] & F_COMMIT and turns[k, T_CALL_START] >= 0]
    if len(commit) > cfg.max_branch_turns:
        commit = sorted(rng.choice(commit, cfg.max_branch_turns, replace=False).tolist())
    spans, corr, anchors = [], [], []
    for k in commit:
        cs, ce, ts = int(turns[k, T_CALL_START]), int(turns[k, T_CALL_END]), int(turns[k, T_START])
        if cs < 1 or ce <= cs or ce > n or ts >= cs:      # malformed / out-of-range span: skip, never crash
            continue
        spans.append((cs, ce, ts, ce))
        corr.append(np.arange(cs - 1, ce - 1))
        anchors.append(cs - 1)
    corr_pos = np.concatenate(corr) if corr else np.zeros(0, np.int64)
    # L_ce: assistant tokens of successful trajectories only (design invariant 7)
    ce_pos = np.zeros(0, np.int64)
    if int(traj["resolved"]) == 1:
        rngs = [np.arange(int(t[T_CE_START]) - 1, int(t[T_CE_END]) - 1) for t in turns if t[T_CE_END] > t[T_CE_START]]
        ce_pos = np.concatenate(rngs) if rngs else ce_pos
    ce_pos = ce_pos[(ce_pos >= 0) & (ce_pos < n - 1)]
    ce_tgt = ids[ce_pos + 1].astype(np.int64)
    # L_keep: non-commit positions, mostly inside assistant turns
    asst = [np.arange(int(t[T_CE_START]) - 1, int(t[T_CE_END]) - 1) for t in turns if t[T_CE_END] > t[T_CE_START]]
    asst = np.setdiff1d(np.concatenate(asst) if asst else np.zeros(0, np.int64), corr_pos)
    asst = asst[(asst >= 0) & (asst < n - 1)]
    other = np.setdiff1d(np.arange(0, n - 1), np.concatenate([corr_pos, asst]))
    n_a = min(len(asst), int(round(cfg.n_keep * cfg.keep_assistant_frac)))
    n_o = min(len(other), cfg.n_keep - n_a)
    keep = np.concatenate([rng.choice(asst, n_a, replace=False) if n_a else asst[:0],
                           rng.choice(other, n_o, replace=False) if n_o else other[:0]]).astype(np.int64)
    plan = Plan(n=n, ce_pos=ce_pos.astype(np.int64), ce_tgt=ce_tgt, spans=spans, corr_pos=corr_pos.astype(np.int64),
                anchor_pos=np.asarray(anchors, np.int64), keep_pos=np.sort(keep))
    plan.H = np.unique(np.concatenate([plan.corr_pos, plan.keep_pos]))
    plan.P = np.unique(np.concatenate([plan.H, plan.ce_pos]))
    return plan


# --------------------------------------------------------------------------- pass A
@torch.no_grad()
def _branch(model, cache, ids_t, hint_t, span, cfg: StepCfg):
    """p_full over one commit span: hint + turn on a forked cache. Returns packed (ids, lp, tail)."""
    cs, ce, ts, cend = span
    fork = cache.fork()
    toks = torch.cat([hint_t, ids_t[ts:cend]])
    s0 = hint_t.numel() + (cs - ts) - 1            # branch index that predicts token `cs`
    s1 = s0 + (ce - cs)
    rows = []
    for s in range(0, toks.numel(), cfg.max_chunk):
        e = min(s + cfg.max_chunk, toks.numel())
        x = model.run_layers(model.embed(toks[None, s:e]), fork, ts + s)[0]
        lo, hi = max(s0, s), min(s1, e)
        if hi > lo:
            rows.append(x[lo - s:hi - s])
    h = model.final_norm(torch.cat(rows))
    out = [pack_topk(model.logits(h[i:i + 128]).float(), cfg.topk) for i in range(0, h.shape[0], 128)]
    return tuple(torch.cat(z) for z in zip(*out))


@torch.no_grad()
def pass_a(model, ids_t, hint_t, plan: Plan, cfg: StepCfg):
    n, n_layers, d = ids_t.shape[0], len(model.layers), model.c.hidden_size
    dev = ids_t.device
    xs = [torch.empty(n, d, dtype=model.dtype, device=dev) for _ in range(n_layers + 1)]
    starts = {sp[2]: k for k, sp in enumerate(plan.spans)}
    cuts = sorted({0, n} | set(starts))
    packed = [None] * len(plan.spans)
    cache = model.new_cache()
    for a, b in zip(cuts[:-1], cuts[1:]):
        if a in starts:
            packed[starts[a]] = _branch(model, cache, ids_t, hint_t, plan.spans[starts[a]], cfg)
        for s in range(a, b, cfg.max_chunk):
            e = min(s + cfg.max_chunk, b)
            model.run_layers(model.embed(ids_t[None, s:e]), cache, s, record=xs)
    return xs, packed


@torch.no_grad()
def compute_targets(model, xs, plan: Plan, packed, cfg: StepCfg):
    """Magnitude target y_k = log1p(sum_t KL(p_full || p_plain)) per sampled turn (+ first-4-token KL)."""
    y, first4 = [], []
    for (cs, ce, _, _), (ids, lp, tail) in zip(plan.spans, packed):
        pos = torch.arange(cs - 1, ce - 1, device=xs[-1].device)
        h = model.final_norm(xs[-1][pos])
        kl = torch.cat([kl_to_packed(model.logits(h[i:i + 128]), ids[i:i + 128], lp[i:i + 128], tail[i:i + 128])
                        for i in range(0, h.shape[0], 128)])
        y.append(torch.log1p(kl.sum()))
        first4.append(kl[:4].sum())
    dev = xs[-1].device
    return (torch.stack(y) if y else torch.zeros(0, device=dev)), (torch.stack(first4) if first4 else torch.zeros(0, device=dev))


# --------------------------------------------------------------------------- heads + losses
def _idx(sorted_arr: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return torch.searchsorted(sorted_arr, q)


def head_stage(model, heads, plan: Plan, packed, y, rows_final, rows_tap, gain: float, cfg: StepCfg, norms: dict):
    """Losses on gathered rows. rows_final: [|P|,d] final-layer output at plan.P (pre final-norm);
    rows_tap(j): [|H|,d] output of tap layer j at plan.H. gain scales the head->backbone gradient
    (0 during warm-up: taps are detached)."""
    dev = rows_final.device
    w = model.lm_head.weight
    P = torch.as_tensor(plan.P, device=dev)
    H = torch.as_tensor(plan.H, device=dev)
    h_all = model.final_norm(rows_final)
    zero = rows_final.new_zeros((), dtype=torch.float32)
    out = {"ce": zero, "corr": zero, "keep": zero, "mag": zero}
    if len(plan.ce_pos):
        idx = _idx(P, torch.as_tensor(plan.ce_pos, device=dev))
        out["ce"] = ce_sum(h_all[idx], torch.as_tensor(plan.ce_tgt, device=dev), w) / max(norms["ce"], 1)
    stats = {}
    if len(plan.H):
        ih = _idx(P, H)
        h_sg = h_all[ih].detach()
        taps = [rows_tap(j) for j in range(len(cfg.tap_fracs))] + [rows_final[ih]]
        taps = [GradScale.apply(t, gain) if gain > 0 else t.detach() for t in taps]
        h_mix = heads.mix(taps)
        delta = heads.delta(h_mix)
        mlog = heads.mag_logits(h_mix)
        g = heads.gate(mlog)
        if len(plan.corr_pos):
            ic = _idx(H, torch.as_tensor(plan.corr_pos, device=dev))
            ids = torch.cat([p[0] for p in packed])
            lp = torch.cat([p[1] for p in packed])
            tail = torch.cat([p[2] for p in packed])
            out["corr"] = corr_sum(h_sg[ic], delta[ic], g[ic], w, ids, lp, tail) / max(norms["corr"], 1)
            stats["gate_corr"] = g[ic].mean().detach()
            ia = _idx(H, torch.as_tensor(plan.anchor_pos, device=dev))
            out["mag"] = heads.hl_gauss_loss(mlog[ia], y).sum() / max(norms["anchor"], 1)
            stats["mag_pred"] = heads.expected_mag(mlog[ia]).mean().detach()
            stats["mag_target"] = y.mean().detach()
        if len(plan.keep_pos):
            ik = _idx(H, torch.as_tensor(plan.keep_pos, device=dev))
            out["keep"] = keep_sum(h_sg[ik], delta[ik], g[ik], w) / max(norms["keep"], 1)
            stats["gate_keep"] = g[ik].mean().detach()
    out["total"] = out["ce"] + cfg.lambda_corr * out["corr"] + cfg.lambda_keep * out["keep"] + cfg.lambda_mag * out["mag"]
    return out, stats


# --------------------------------------------------------------------------- one trajectory
def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()


def run_trajectory(model, heads, traj: dict, plan: Plan, cfg: StepCfg, norms: dict, gain: float,
                   train: bool = True) -> dict:
    dev = model.embed_tokens.weight.device
    ids_t = torch.as_tensor(np.asarray(traj["ids"], dtype=np.int64), device=dev)
    hint_t = torch.as_tensor(np.asarray(traj["hint"], dtype=np.int64), device=dev)
    n_layers = len(model.layers)
    t0 = _sync(dev)
    with prof.rf("p:pass_a"):
        xs, packed = pass_a(model, ids_t, hint_t, plan, cfg)
        y, first4 = compute_targets(model, xs, plan, packed, cfg)
    t1 = _sync(dev)
    P = torch.as_tensor(plan.P, device=dev)
    H = torch.as_tensor(plan.H, device=dev)
    taps_i = tap_indices(n_layers, cfg.tap_fracs)
    want_grad = train
    rows_final = xs[n_layers][P].detach().requires_grad_(want_grad)
    taps = [xs[i][H].detach().requires_grad_(want_grad and gain > 0) for i in taps_i]
    with prof.rf("p:heads"), torch.set_grad_enabled(want_grad):
        losses, stats = head_stage(model, heads, plan, packed, y, rows_final, lambda j: taps[j], gain, cfg, norms)
    t2 = _sync(dev)
    metrics = {
        "loss_ce": float(losses["ce"].detach()), "loss_corr": float(losses["corr"].detach()), "loss_keep": float(losses["keep"].detach()),
        "loss_mag": float(losses["mag"].detach()), "loss": float(losses["total"].detach()), "tokens": plan.n,
        **{f"n_{k}": v for k, v in plan.counts.items()}, "n_turns": len(plan.spans),
        "kl_first4_mean": float(first4.mean()) if len(first4) else 0.0,
        "y_mean": float(y.mean()) if len(y) else 0.0,
        **{k: float(v) for k, v in stats.items()},
        "t_pass_a": t1 - t0, "t_heads": t2 - t1,
    }
    if not train:
        return metrics
    with prof.rf("p:heads"):
        if losses["total"].requires_grad:
            losses["total"].backward()
    g = torch.zeros(plan.n, model.c.hidden_size, dtype=model.dtype, device=dev)
    if rows_final.grad is not None:
        g.index_add_(0, P, rows_final.grad.to(g.dtype))
    tap_grads = {}
    if gain > 0:
        for j, i in enumerate(taps_i):
            if taps[j].grad is not None:
                tap_grads.setdefault(i, []).append(taps[j].grad.to(g.dtype))
    xs[n_layers] = None
    _rev = prof.rf("p:reverse")
    _rev.__enter__()
    for tg in tap_grads.get(n_layers, []):              # a tap on the final output itself
        g.index_add_(0, H, tg)
    for l in reversed(range(n_layers)):
        if l + 1 < n_layers:                            # g = d/d xs[l+1]: add direct head grads of taps there
            for tg in tap_grads.get(l + 1, []):
                g.index_add_(0, H, tg)
        x_in = xs[l].detach().requires_grad_(l > 0)
        with torch.enable_grad():
            out = model.layers[l](x_in[None], None, 0)[0]
        out.backward(g)
        g = x_in.grad if l > 0 else None
        xs[l] = None
        del out, x_in
    _rev.__exit__(None, None, None)
    t3 = _sync(dev)
    metrics["t_reverse"] = t3 - t2
    if dev.type == "cuda":
        metrics["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
    return metrics
