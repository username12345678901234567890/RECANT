from __future__ import annotations
import gc
import math
import signal
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
import numpy as np

def _is_oom(e: BaseException, torch) -> bool:
    return isinstance(e, torch.cuda.OutOfMemoryError) or 'out of memory' in str(e).lower()

def empty_hint_ids(store, model_path: Path):
    import numpy as np
    tok = model_path / 'tokenizer.json'
    if tok.exists():
        from tokenizers import Tokenizer
        from .data.render import Renderer
        return Renderer(Tokenizer.from_file(str(tok))).render_hint('')
    h = np.asarray(store.get(0)['hint'])
    return np.concatenate([h[:2], h[-2:]]).astype(np.uint32)

def calibrate(model, store, indices, a, cfg, empty_hint, rng, log):
    import torch
    with torch.no_grad():
        return _calibrate(model, store, indices, a, cfg, empty_hint, rng, log)

def _calibrate(model, store, indices, a, cfg, empty_hint, rng, log):
    import torch
    from .branch import compute_targets, pass_a, plan_trajectory
    dev = model.embed_tokens.weight.device
    ys, floors = ([], [])
    ccfg = replace(cfg, max_branch_turns=min(cfg.max_branch_turns, a.calib_turns))
    for i in indices[:a.calib_trajs]:
        t = store.get(i)
        plan = plan_trajectory(t, ccfg, rng)
        if not plan.spans:
            continue
        ids_t = torch.as_tensor(np.asarray(t['ids'], dtype=np.int64), device=dev)
        for hint, sink in ((np.asarray(t['hint']), ys), (np.asarray(empty_hint), floors)):
            xs, packed = pass_a(model, ids_t, torch.as_tensor(hint.astype(np.int64), device=dev), plan, ccfg)
            y, _ = compute_targets(model, xs, plan, packed, ccfg)
            sink += y.tolist()
            del xs
        gc.collect()
    q = lambda v, p: float(np.percentile(v, p)) if len(v) else None
    y_max = max(2.0, (q(ys, 99.5) or 0.0) * 1.1) if ys else a.y_max
    rep = {'n_turns': len(ys), 'y_real_pct': {p: q(ys, p) for p in (10, 50, 90, 99, 99.5)}, 'y_empty_hint_pct': {p: q(floors, p) for p in (10, 50, 90, 99)}, 'y_max': y_max}
    log(f'[calib] {rep}')
    return rep

def run(a) -> dict:
    t_start = time.time()
    from .env import choose_tmp_dir, find_root, install_wheels, probe_env, resolve_input, setup_process_env, write_json
    tmp = choose_tmp_dir(a.tmp_dir, min_free_gb=a.min_tmp_gb)
    setup_process_env(tmp)
    out_dir = Path(a.out_dir)
    from .runlog import RunLog, system_metrics
    rl = RunLog(out_dir)
    log = lambda *m: print(*m, flush=True)
    env = probe_env(extra_paths=(str(out_dir), a.data_dir, a.model_path))
    write_json(out_dir / 'env.json', {'env': env, 'args': vars(a), 'scratch': str(tmp)})
    log(f"[train] scratch {tmp}; out {out_dir}; disks {env['disks']}")
    if a.wheel_dir:
        installed = install_wheels(resolve_input(a.wheel_dir, tmp, 'wheel_dir'), tmp, log=log)
        log(f'[train] installed wheels: {installed}')
    import torch
    from . import backends
    from .branch import StepCfg, plan_trajectory, run_trajectory
    from .ckpt import Shadow, save_final, smoke_test_save
    from .data.packer import MemoryModel, StepPlanner
    from .data.prep import file_sha256
    from .data.store import TrajectoryStore
    from .heads import Heads
    from .model.config import TextConfig
    from .model.qwen35 import load_hf
    from .sched import HeadGain, TimeBudget, lr_at
    budget = TimeBudget(a.time_limit_hours, a.save_reserve_min, t0=t_start)
    device = a.device
    dtype = {'bf16': torch.bfloat16, 'fp32': torch.float32}[a.dtype]
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    data_dir = find_root(resolve_input(a.data_dir, tmp, 'data_dir'), 'manifest.json')
    store = TrajectoryStore(data_dir)
    model_path = find_root(resolve_input(a.model_path, tmp, 'model_path'), 'config.json')
    tok_json = model_path / 'tokenizer.json'
    want = store.manifest.get('tokenizer_sha256')
    if want and tok_json.exists() and (file_sha256(tok_json) != want):
        raise RuntimeError('tokenizer.json in --model_path differs from the one the dataset was tokenized with')
    if want and (not tok_json.exists()):
        log("[train] WARNING: no tokenizer.json in model_path; cannot verify the dataset's tokenizer hash")
    train_idx = store.split_indices('train')
    val_idx = store.split_indices('val')
    if not val_idx:
        log('[train] WARNING: no validation split in the dataset; evaluating on training trajectories')
        val_idx = train_idx[:a.eval_trajs]
    log(f"[train] dataset: {len(store)} trajectories, {store.manifest['total_tokens'] / 1000000.0:.1f}M tokens ({len(train_idx)} train / {len(val_idx)} val)")
    cfg_m = TextConfig.from_hf(model_path)
    from . import fast
    from .profile_util import StepProfiler
    fast.configure(a.no_fast, a.fast_disable)
    profile_steps = {int(x) for x in str(a.profile_steps).split(',') if x.strip()}
    rep = backends.select_all(device, cfg_m, log)
    rep['fast_paths'] = backends.select_fast_paths(device, cfg_m, log)
    write_json(out_dir / 'backends.json', rep)
    quantize = not a.no_quantize
    model = load_hf(model_path, device=device, dtype=dtype, quantize=quantize, cfg=cfg_m, log=log)
    n_lora = model.add_lora(a.lora_r, a.lora_alpha)
    lora_params = model.lora_parameters()
    n_params = sum((p.numel() for p in lora_params))
    step_cfg = StepCfg(lambda_corr=a.lambda_corr, lambda_keep=a.lambda_keep, lambda_mag=a.lambda_mag, max_branch_turns=a.max_branch_turns, n_keep=a.n_keep, topk=a.topk, max_chunk=a.max_chunk)
    heads = Heads(cfg_m.hidden_size, n_taps=len(step_cfg.tap_fracs) + 1, hidden=a.head_hidden, n_bins=a.n_bins, y_max=a.y_max).to(device)
    log(f'[train] LoRA r={a.lora_r} on {n_lora} linears = {n_params / 1000000.0:.1f}M params; heads {sum((p.numel() for p in heads.parameters())) / 1000000.0:.1f}M params; FP4 backbone={quantize}')
    smoke_test_save(tmp, model, heads, log)
    calib = {}
    if a.calib_trajs > 0:
        calib = calibrate(model, store, val_idx or train_idx, a, step_cfg, empty_hint_ids(store, model_path), rng, log)
        heads.set_range(calib['y_max'], reset_gate=True)
    write_json(out_dir / 'calibration.json', calib)
    groups = [{'params': lora_params, 'base_lr': a.lr_lora}, {'params': list(heads.parameters()), 'base_lr': a.lr_heads}]
    opt = torch.optim.AdamW([dict(g, lr=g['base_lr']) for g in groups], betas=(0.9, 0.95), weight_decay=0.0)
    all_params = lora_params + list(heads.parameters())
    if torch.device(device).type == 'cuda':
        total_mem = torch.cuda.get_device_properties(device).total_memory
        static = torch.cuda.memory_allocated() + 4 * n_params * 4 * 1.0
    else:
        total_mem, static = (float('inf'), 0.0)
    itemsize = torch.tensor([], dtype=dtype).element_size()
    prior = a.bytes_per_token_prior or ((cfg_m.num_hidden_layers + 1) * cfg_m.hidden_size + 8 * cfg_m.intermediate_size) * itemsize
    mem = MemoryModel(static_bytes=static, bytes_per_token=prior)
    lengths = store.index['length']
    planner = StepPlanner(train_idx, lengths, a.tokens_per_step, mem, a.vram_target * total_mem, max_len=int(store.manifest.get('max_seq_len', max(lengths))), seed=a.seed)
    gain_fn = HeadGain(a.head_warmup_steps, a.head_warmup_frac, a.head_gain_ramp, a.head_gain)
    shadow = Shadow()
    stop = {'flag': False}
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, lambda *_: stop.__setitem__('flag', True))
        except ValueError:
            pass
    state = {'step': 0, 'tokens': 0, 'trajectories': 0, 'ooms': 0, 'nonfinite_skips': 0}
    meta = {'config': {'args': vars(a), 'model': cfg_m.__dict__, 'tap_indices': None, 'heads': {'n_taps': heads.n_taps, 'hidden': a.head_hidden, 'n_bins': a.n_bins, 'y_max': heads.y_max}, 'lora': {'r': a.lora_r, 'alpha': a.lora_alpha}, 'fp4_backbone': quantize, 'fp4_gemm': rep['fp4_gemm']['chosen'], 'gdn': rep['gdn']['chosen'], 'calibration': calib, 'dataset_manifest': {k: v for k, v in store.manifest.items() if k not in ('config', 'hint_template')}}, 'state': state}
    from .branch import tap_indices
    meta['config']['tap_indices'] = tap_indices(len(model.layers), step_cfg.tap_fracs)

    def snapshot_grads():
        return [None if p.grad is None else p.grad.clone() for p in all_params]

    def restore_grads(snap):
        for p, g in zip(all_params, snap):
            p.grad = g

    def guarded(traj, plan, norms, gain):
        snap = snapshot_grads() if any((p.grad is not None for p in all_params)) else None
        cfg = step_cfg
        for attempt in (0, 1):
            oomed = False
            try:
                if torch.device(device).type == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
                m = run_trajectory(model, heads, traj, plan, cfg, norms, gain, train=True)
                planner.report_ok(plan.n, m.get('peak_mem_gb', 0) * 2 ** 30 if 'peak_mem_gb' in m else None)
                return m
            except Exception as e:
                if not _is_oom(e, torch):
                    raise
                oomed = True
            if oomed:
                state['ooms'] += 1
                planner.report_oom(plan.n)
                for p in all_params:
                    p.grad = None
                if snap is not None:
                    restore_grads(snap)
                gc.collect()
                if torch.device(device).type == 'cuda':
                    torch.cuda.empty_cache()
                log(f'[train] OOM at {plan.n} tokens (attempt {attempt}); cap now {planner.token_cap:.0f}')
                if attempt == 0:
                    cfg = replace(step_cfg, max_branch_turns=max(1, step_cfg.max_branch_turns // 2), max_chunk=max(2048, step_cfg.max_chunk // 2))
                    plan = plan_trajectory(traj, cfg, rng)
        return None

    def evaluate(gain):
        acc: dict = defaultdict(list)
        er = np.random.default_rng(1234)
        for i in val_idx[:a.eval_trajs]:
            t = store.get(i)
            plan = plan_trajectory(t, step_cfg, er)
            norms = {k: max(v, 1) for k, v in plan.counts.items()}
            try:
                m = run_trajectory(model, heads, t, plan, step_cfg, norms, gain, train=False)
            except Exception as e:
                if _is_oom(e, torch):
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
                raise
            for k, v in m.items():
                acc[k].append(v)
        res = {k: float(np.mean(v)) for k, v in acc.items()}
        if 'gate_corr' in res and 'gate_keep' in res:
            res['gate_separation'] = res['gate_corr'] - res['gate_keep']
        return res
    budget.start_training()
    shadow.update(model, heads, state)
    t_last_eval, nonfinite_run = (time.time(), 0)
    log(f'[train] setup took {time.time() - t_start:.0f}s; training window {budget.remaining() / 3600:.2f} h')
    try:
        while not budget.done() and (not stop['flag']) and (a.max_steps is None or state['step'] < a.max_steps):
            frac = budget.frac()
            mult = lr_at(state['step'], frac, 1.0, a.warmup_steps, a.min_lr_ratio)
            for g in opt.param_groups:
                g['lr'] = g['base_lr'] * mult
            gain = gain_fn(state['step'], frac)
            ids = planner.next_step()
            if not ids:
                log('[train] no trajectory fits the memory budget / data exhausted; stopping')
                break
            trajs = [store.get(i) for i in ids]
            plans = [plan_trajectory(t, step_cfg, rng) for t in trajs]
            norms = {k: sum((p.counts[k] for p in plans)) for k in ('ce', 'corr', 'keep', 'anchor')}
            t_step = time.time()
            agg, n_ok, tok = (defaultdict(float), 0, 0)
            prof_cm = StepProfiler(out_dir, state['step'] + 1, log) if state['step'] + 1 in profile_steps else None
            if prof_cm:
                prof_cm.__enter__()
            for t, plan in zip(trajs, plans):
                m = guarded(t, plan, norms, gain)
                if m is None:
                    continue
                n_ok += 1
                tok += plan.n
                for k, v in m.items():
                    if k.startswith(('loss', 't_')) or k in ('n_ce', 'n_corr', 'n_keep', 'n_turns'):
                        agg[k] += v
                    elif k in ('gate_corr', 'gate_keep', 'mag_pred', 'mag_target', 'kl_first4_mean', 'peak_mem_gb'):
                        agg[k] = max(agg[k], v) if k == 'peak_mem_gb' else agg[k] + v / max(len(trajs), 1)
                if budget.done() or stop['flag']:
                    break
            if prof_cm:
                prof_cm.__exit__(None, None, None)
            gnorm = float(torch.nn.utils.clip_grad_norm_(all_params, a.grad_clip)) if n_ok else float('nan')
            if n_ok and math.isfinite(gnorm) and math.isfinite(agg['loss']):
                opt.step()
                nonfinite_run = 0
                planner.end_step_clean()
            else:
                state['nonfinite_skips'] += 1
                nonfinite_run += 1
                if nonfinite_run >= 20:
                    raise RuntimeError('20 consecutive non-finite steps; aborting')
            opt.zero_grad(set_to_none=True)
            state['step'] += 1
            state['tokens'] += tok
            state['trajectories'] += n_ok
            dt = time.time() - t_step
            rec = {'step': state['step'], 'epoch': planner.epoch, 'frac': round(frac, 4), 'lr_lora': opt.param_groups[0]['lr'], 'lr_heads': opt.param_groups[1]['lr'], 'head_gain': gain, 'n_traj': n_ok, 'tokens': tok, 'tokens_total': state['tokens'], 'tok_per_s': tok / max(dt, 1e-09), 'step_s': dt, 'grad_norm': gnorm, 'ooms': state['ooms'], 'skipped_memory': planner.skipped_memory, 'token_cap': planner.token_cap, 'mem_c0_gb': mem.c0 / 2 ** 30, 'mem_c1_mb_per_tok': mem.c1 / 2 ** 20, 'elapsed_h': budget.elapsed() / 3600, **dict(agg)}
            if state['step'] % a.sys_every == 0:
                rec.update(system_metrics())
            rl.jsonl('train', rec)
            if state['step'] % a.log_every == 0:
                log(f"[train] step {state['step']} loss {agg['loss']:.4f} (ce {agg['loss_ce']:.3f} corr {agg['loss_corr']:.3f} keep {agg['loss_keep']:.3f} mag {agg['loss_mag']:.3f}) gain {gain:.3f} {rec['tok_per_s']:.0f} tok/s frac {frac:.3f} tokens {state['tokens'] / 1000000.0:.2f}M")
            if time.time() - shadow.at > a.shadow_every_min * 60:
                shadow.update(model, heads, state)
            if a.eval_every_min > 0 and time.time() - t_last_eval > a.eval_every_min * 60:
                t_last_eval = time.time()
                rl.jsonl('eval', {'step': state['step'], 'tokens_total': state['tokens'], **evaluate(gain)})
        final_eval = {}
        if a.final_eval and (not stop['flag']) and (time.time() < budget.hard_deadline - 120):
            final_eval = evaluate(gain_fn(state['step'], 1.0))
            rl.jsonl('eval', {'step': state['step'], 'tokens_total': state['tokens'], 'final': True, **final_eval})
        status = 'stopped_by_signal' if stop['flag'] else 'time_limit' if budget.done() else 'finished'
    except BaseException as e:
        status = f'error: {type(e).__name__}: {e}'
        log(f'[train] ERROR {status}')
        import traceback
        traceback.print_exc()
        final_eval = {}
        err = e
    else:
        err = None
    finally:
        state['status'] = status
        state['elapsed_h'] = budget.elapsed() / 3600
        try:
            save_final(out_dir, model.lora_state_dict(), heads.state_dict(), meta, log)
        except Exception as e2:
            log(f'[train] direct save failed ({e2!r}); saving the host-RAM shadow copy')
            shadow.save(out_dir, meta, log)
        write_json(out_dir / 'summary.json', {'status': status, 'state': state, 'final_eval': final_eval, 'backends': rep, 'calibration': calib, 'seconds_total': time.time() - t_start})
        rl.close()
    if err is not None:
        raise err
    return state