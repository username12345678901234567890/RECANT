import numpy as np
import pytest
import torch

from recant.branch import StepCfg, head_stage, pass_a, plan_trajectory, run_trajectory, tap_indices
from recant.data.render import (F_COMMIT, F_REASONING, T_ANCHOR, T_CALL_END, T_CALL_START, T_CE_END, T_CE_START,
                                T_FLAGS, T_NCALLS, T_START, TURN_STRIDE)
from recant.heads import Heads
from recant.losses import pack_topk
from recant.model.config import TextConfig
from recant.model.qwen35 import Qwen35

V = 300


def make_model(seed=0, lora=True):
    torch.manual_seed(seed)
    m = Qwen35(TextConfig.tiny(vocab_size=V, layers=4), device="cpu", dtype=torch.float32)
    m.init_random(std=0.05, seed=seed)
    if lora:
        m.add_lora(r=4, alpha=4)
        with torch.no_grad():                       # non-zero B so LoRA grads are non-trivial
            for n, p in m.named_parameters():
                if "lora_B" in n:
                    p.normal_(0, 0.02)
    return m


def make_traj(resolved=1, n=150, seed=0, commits=(1, 2)):
    rng = np.random.default_rng(seed)
    ids = rng.integers(3, V, n).astype(np.uint32)
    turns = np.full((3, TURN_STRIDE), -1, np.int32)
    for k, (s, e) in enumerate([(12, 42), (55, 95), (105, 140)]):
        turns[k, T_START], turns[k, T_CE_START], turns[k, T_CE_END] = s, s + 3, e
        turns[k, T_CALL_START], turns[k, T_CALL_END] = s + 12, e - 2
        turns[k, T_ANCHOR] = s + 11
        turns[k, T_FLAGS] = (F_COMMIT if k in commits else 0) | F_REASONING
        turns[k, T_NCALLS] = 1
    hint = rng.integers(3, V, 7).astype(np.uint32)
    return {"ids": ids, "hint": hint, "turns": turns, "resolved": resolved}


def norms_for(plan):
    return {k: max(v, 1) for k, v in plan.counts.items()}


def small_cfg(**kw):
    return StepCfg(topk=32, n_keep=16, max_chunk=40, **kw)


def test_plan_shapes_and_invariants():
    cfg = small_cfg()
    t = make_traj(resolved=0)
    p = plan_trajectory(t, cfg, np.random.default_rng(0))
    assert len(p.ce_pos) == 0                                     # no CE on failed trajectories (invariant 7)
    assert len(p.spans) == 2 and len(p.anchor_pos) == 2
    turns = t["turns"]
    for (cs, ce, ts, _), a in zip(p.spans, p.anchor_pos):
        assert a == cs - 1 and ts < cs
    # keep positions never overlap commit spans (invariant 5)
    assert not set(p.keep_pos.tolist()) & set(p.corr_pos.tolist())
    p1 = plan_trajectory(make_traj(resolved=1), cfg, np.random.default_rng(0))
    assert len(p1.ce_pos) > 0 and (p1.ce_pos < p1.n - 1).all()


def test_pass_a_main_stream_is_unaffected_by_branches():
    m = make_model()
    t = make_traj()
    cfg = small_cfg()
    plan = plan_trajectory(t, cfg, np.random.default_rng(0))
    ids = torch.as_tensor(t["ids"].astype(np.int64))
    hint = torch.as_tensor(t["hint"].astype(np.int64))
    xs, packed = pass_a(m, ids, hint, plan, cfg)
    with torch.no_grad():
        x = m.embed(ids[None])
        for i, layer in enumerate(m.layers):
            assert torch.allclose(xs[i], x[0], atol=1e-4, rtol=1e-3), f"layer input {i}"
            x = layer(x)
    assert torch.allclose(xs[-1], x[0], atol=1e-4, rtol=1e-3)


def test_p_full_equals_bruteforce_forward_on_hinted_sequence():
    m = make_model()
    t = make_traj()
    cfg = StepCfg(topk=64, n_keep=8, max_chunk=33)
    plan = plan_trajectory(t, cfg, np.random.default_rng(0))
    ids = torch.as_tensor(t["ids"].astype(np.int64))
    hint = torch.as_tensor(t["hint"].astype(np.int64))
    _, packed = pass_a(m, ids, hint, plan, cfg)
    for (cs, ce, ts, cend), (pid, plp, ptail) in zip(plan.spans, packed):
        full = torch.cat([ids[:ts], hint, ids[ts:cend]])[None]
        logits = m.forward_logits(full)[0]
        s0 = ts + len(hint) + (cs - ts) - 1
        bid, blp, btail = pack_topk(logits[s0:s0 + (ce - cs)], 64)
        assert torch.equal(pid, bid)
        assert torch.allclose(plp, blp, atol=2e-4), (plp - blp).abs().max()
        assert torch.allclose(ptail, btail, atol=2e-3)


def _reference_grads(m, heads, t, plan, cfg, gain, norms):
    """Same losses through ordinary autograd on a full forward (no manual backward)."""
    ids = torch.as_tensor(t["ids"].astype(np.int64))
    hint = torch.as_tensor(t["hint"].astype(np.int64))
    with torch.no_grad():
        xs0, packed = pass_a(m, ids, hint, plan, cfg)
        from recant.branch import compute_targets
        y, _ = compute_targets(m, xs0, plan, packed, cfg)
    x = m.embed(ids[None])
    conn = [x[0]]
    for layer in m.layers:
        x = layer(x)
        conn.append(x[0])
    P, H = torch.as_tensor(plan.P), torch.as_tensor(plan.H)
    taps_i = tap_indices(len(m.layers), cfg.tap_fracs)
    out, _ = head_stage(m, heads, plan, packed, y, conn[-1][P], lambda j: conn[taps_i[j]][H], gain, cfg, norms)
    out["total"].backward()


def _grads(module_params):
    return {n: (None if p.grad is None else p.grad.clone()) for n, p in module_params}


def _zero(m, heads):
    for p in list(m.parameters()) + list(heads.parameters()):
        p.grad = None


@pytest.mark.parametrize("resolved,gain", [(1, 0.0), (1, 1.0), (0, 1.0)])
def test_manual_reverse_pass_matches_autograd(resolved, gain):
    m = make_model()
    heads = Heads(m.c.hidden_size, hidden=16, n_bins=11, y_max=6.0)
    with torch.no_grad():                            # make the correction head non-trivial
        heads.delta_mlp[2].weight.normal_(0, 0.05)
    t = make_traj(resolved=resolved)
    cfg = small_cfg()
    plan = plan_trajectory(t, cfg, np.random.default_rng(1))
    norms = norms_for(plan)
    run_trajectory(m, heads, t, plan, cfg, norms, gain, train=True)
    got_lora = _grads(m.named_parameters())
    got_head = _grads(heads.named_parameters())
    _zero(m, heads)
    _reference_grads(m, heads, t, plan, cfg, gain, norms)
    ref_lora = _grads(m.named_parameters())
    ref_head = _grads(heads.named_parameters())
    checked = 0
    for name, ref in {**ref_lora, **ref_head}.items():
        got = {**got_lora, **got_head}[name]
        if ref is None:
            assert got is None or got.abs().max() == 0, name
            continue
        assert got is not None, f"missing grad {name}"
        scale = ref.abs().max().clamp(min=1e-8)
        assert (got - ref).abs().max() / scale < 2e-2, (name, ((got - ref).abs().max() / scale).item())
        checked += 1
    assert checked > 10


def test_invariant_backbone_gets_no_gradient_from_corr_or_keep_via_h_L():
    """Warm-up (gain 0) + a failed trajectory (no CE): only L_corr/L_keep/L_mag act. Since h_L is
    stop-grad in L_corr/L_keep and taps are detached, the LoRA branch must receive exactly zero."""
    m = make_model()
    heads = Heads(m.c.hidden_size, hidden=16, n_bins=11, y_max=6.0)
    with torch.no_grad():
        heads.delta_mlp[2].weight.normal_(0, 0.05)
    t = make_traj(resolved=0)
    cfg = small_cfg()
    plan = plan_trajectory(t, cfg, np.random.default_rng(2))
    r = run_trajectory(m, heads, t, plan, cfg, norms_for(plan), gain=0.0, train=True)
    assert r["loss_corr"] > 0 and r["loss_keep"] >= 0
    for n, p in m.named_parameters():
        if "lora_" in n:
            assert p.grad is None or p.grad.abs().max() == 0, n
    assert heads.delta_mlp[2].weight.grad.abs().max() > 0        # ... while the heads do learn


def test_invariant_gate_input_is_stop_grad():
    """The magnitude head is trained by L_mag alone; L_corr/L_keep reach the gate scalars a, b only."""
    m = make_model()
    heads = Heads(m.c.hidden_size, hidden=16, n_bins=11, y_max=6.0)
    with torch.no_grad():
        heads.delta_mlp[2].weight.normal_(0, 0.05)
    t = make_traj(resolved=0)
    cfg = small_cfg(lambda_mag=0.0)
    plan = plan_trajectory(t, cfg, np.random.default_rng(3))
    run_trajectory(m, heads, t, plan, cfg, norms_for(plan), gain=0.0, train=True)
    for p in heads.mag_mlp.parameters():
        assert p.grad is None or p.grad.abs().max() == 0
    assert heads.gate_a.grad is not None and heads.gate_b.grad.abs() > 0


def test_targets_do_not_depend_on_heads():
    """p_full / p_plain are computed without any correction (invariants 1 and 2)."""
    m = make_model()
    t = make_traj()
    cfg = small_cfg()
    plan = plan_trajectory(t, cfg, np.random.default_rng(4))
    ids = torch.as_tensor(t["ids"].astype(np.int64))
    hint = torch.as_tensor(t["hint"].astype(np.int64))
    _, p1 = pass_a(m, ids, hint, plan, cfg)
    heads = Heads(m.c.hidden_size, hidden=16, n_bins=11)
    with torch.no_grad():
        for p in heads.parameters():
            p.add_(torch.randn_like(p))
    _, p2 = pass_a(m, ids, hint, plan, cfg)
    for a, b in zip(p1, p2):
        assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_eval_mode_leaves_no_grads_and_reports_metrics():
    m = make_model()
    heads = Heads(m.c.hidden_size, hidden=16, n_bins=11, y_max=6.0)
    t = make_traj()
    cfg = small_cfg()
    plan = plan_trajectory(t, cfg, np.random.default_rng(5))
    r = run_trajectory(m, heads, t, plan, cfg, norms_for(plan), gain=1.0, train=False)
    assert all(p.grad is None for p in list(m.parameters()) + list(heads.parameters()))
    assert {"loss_ce", "loss_corr", "loss_keep", "loss_mag", "gate_corr", "gate_keep", "kl_first4_mean"} <= set(r)


def test_out_of_range_spans_are_skipped_not_fatal():
    t = make_traj(n=100)                       # turn 3 (105..140) lies beyond the sequence
    plan = plan_trajectory(t, small_cfg(), np.random.default_rng(0))
    assert all(ce <= plan.n for _, ce, _, _ in plan.spans) and len(plan.spans) == 1
