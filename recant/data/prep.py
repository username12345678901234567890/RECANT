from __future__ import annotations
import hashlib
import os
import time
import zipfile
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
import numpy as np
from .hint import DEFAULT_TEMPLATE, build_hint
from .render import F_COMMIT, F_FINISH, F_REASONING, T_CE_END, T_FLAGS, TURN_STRIDE, RenderError, Renderer, verify_against_hf
from .writer import ShardWriter
REPO_ID = 'nvidia/Open-SWE-Traces'
TOKENIZER_FILES = ['tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'vocab.json', 'merges.txt']
COLUMNS = ['instance_id', 'repo', 'trajectory_id', 'language', 'resolved', 'messages', 'tools', 'metadata.reference_patch.patch']
LEN_BINS = [16384, 32768, 49152, 65536, 98304, 131072]
LEN_LABELS = ['<=16k', '<=32k', '<=48k', '<=64k', '<=96k', '<=128k', '>128k']
_W: dict = {}

def parse_source(path: str) -> dict | None:
    p = path.split('/')
    if len(p) != 5 or p[0] != 'data' or (not p[4].endswith('.parquet')):
        return None
    return {'path': path, 'harness': p[1], 'model': p[2], 'source': p[3]}

def source_key(j: dict) -> str:
    return f"{j['harness']}/{j['model']}/{j['source']}"

def split_of(instance_id: str, val_pct: int) -> str:
    h = int(hashlib.md5(instance_id.encode()).hexdigest()[:8], 16)
    return 'val' if h % 100 < val_pct else 'train'

def ensure_tokenizer(spec: str, cache_dir: Path) -> Path:
    if os.path.isdir(spec):
        return Path(spec)
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(spec, allow_patterns=TOKENIZER_FILES, cache_dir=str(cache_dir)))

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def truncate_rendered(ids: np.ndarray, turns: np.ndarray, max_len: int):
    ok = np.nonzero(turns[:, T_CE_END] <= max_len)[0]
    if len(ok) == 0:
        return None
    k = int(ok[-1])
    return (ids[:int(turns[k, T_CE_END])], turns[:k + 1])

def _init_worker(cfg: dict) -> None:
    os.environ['RAYON_NUM_THREADS'] = str(cfg['rayon_threads'])
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'true')
    for k, v in cfg['env'].items():
        os.environ.setdefault(k, v)
    from tokenizers import Tokenizer
    _W['cfg'] = cfg
    _W['renderer'] = Renderer(Tokenizer.from_file(cfg['tokenizer_json']))

def process_file(job: dict) -> dict:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    cfg, R = (_W['cfg'], _W['renderer'])
    stats: Counter = Counter()
    recs: list[dict] = []
    raw: list[dict] = []
    local = hf_hub_download(REPO_ID, job['path'], repo_type='dataset', cache_dir=cfg['hf_cache'])
    max_len = cfg['max_seq_len']
    t0 = time.time()
    try:
        pf = pq.ParquetFile(local)
        limit = job.get('max_rows')
        for batch in pf.iter_batches(batch_size=128, columns=COLUMNS):
            stats['rows_seen'] += batch.num_rows
            is_py = pc.equal(batch.column('language'), 'python')
            stats['python_rows'] += int(pc.sum(is_py).as_py() or 0)
            keep = pc.and_(is_py, pc.is_in(batch.column('resolved'), value_set=pa.array(cfg['resolved_values'], pa.int32())))
            batch = batch.filter(keep)
            for r in batch.to_pylist():
                stats['python_resolved01'] += 1
                if limit is not None and stats['python_resolved01'] > limit:
                    break
                _process_row(r, job, cfg, R, max_len, stats, recs, raw)
            if limit is not None and stats['python_resolved01'] > limit:
                break
    finally:
        try:
            real = os.path.realpath(local)
            os.remove(local)
            if real != local and os.path.exists(real):
                os.remove(real)
        except OSError:
            pass
    stats['seconds'] = round(time.time() - t0, 1)
    return {'job': job, 'stats': dict(stats), 'recs': recs, 'raw': raw}

def _process_row(r: dict, job: dict, cfg: dict, R: Renderer, max_len: int, stats: Counter, recs: list, raw: list) -> None:
    msgs = r['messages'] or []
    chars = sum((len(m.get('content') or '') + len(m.get('reasoning_content') or '') for m in msgs))
    if chars > 16 * max_len:
        stats['drop_prefilter_len'] += 1
        return
    try:
        x = R.render(msgs, r['tools'], job['harness'])
    except RenderError as e:
        stats[f'drop_render_{e}'] += 1
        return
    except Exception:
        stats['drop_render_error'] += 1
        return
    n = len(x.ids)
    stats[f"len_{LEN_LABELS[int(np.searchsorted(LEN_BINS, n, side='left'))]}"] += 1
    for name in x.unknown_tools:
        stats[f'unknown_tool:{name}'] += 1
    ids, turns, truncated = (x.ids, x.turns, False)
    if n > max_len:
        if cfg['overlength'] == 'drop':
            stats['drop_overlength'] += 1
            return
        cut = truncate_rendered(ids, turns, max_len)
        if cut is None:
            stats['drop_overlength'] += 1
            return
        ids, turns, truncated = (cut[0], cut[1], True)
        stats['truncated'] += 1
    flags = turns[:, T_FLAGS]
    n_commit = int((flags & F_COMMIT > 0).sum())
    if n_commit == 0:
        stats['drop_no_commit'] += 1
        return
    patch = ((r.get('metadata') or {}).get('reference_patch') or {}).get('patch')
    hint_text = build_hint(patch, cfg['hint_template'])
    if hint_text is None:
        stats['drop_no_hint'] += 1
        return
    stats['kept'] += 1
    recs.append({'ids': ids, 'turns': turns, 'hint': R.render_hint(hint_text), 'meta': {'traj_id': r['trajectory_id'], 'instance_id': r['instance_id'], 'repo': r['repo'], 'harness': job['harness'], 'model': job['model'], 'source': job['source'], 'resolved': int(r['resolved']), 'split': split_of(r['instance_id'], cfg['val_pct']), 'n_turns': len(turns), 'n_commit': n_commit, 'n_finish': int((flags & F_FINISH > 0).sum()), 'n_reasoning_turns': int((flags & F_REASONING > 0).sum()), 'truncated': truncated}})
    if len(raw) < job.get('keep_raw', 0):
        r = dict(r)
        r['_harness'] = job['harness']
        raw.append(r)

class Admission:

    def __init__(self, a):
        self.a = a
        self.inst: Counter = Counter()
        self.repo: Counter = Counter()
        self.source: Counter = Counter()
        self.cls: Counter = Counter()
        self.total = 0
        self.n_active = 1000000
        self.rejects: Counter = Counter()

    @property
    def source_cap_frac(self) -> float:
        return max(self.a.source_frac, 1.05 / max(1, self.n_active))

    @property
    def done(self) -> bool:
        return self.total >= 0.995 * self.a.token_budget

    def try_admit(self, rec: dict) -> bool:
        m, n, a = (rec['meta'], len(rec['ids']), self.a)
        skey = f"{m['harness']}/{m['model']}/{m['source']}"
        if self.total + n > a.token_budget:
            reason = 'budget'
        elif self.inst[m['instance_id']] >= a.max_per_instance:
            reason = 'instance_cap'
        elif self.repo[m['repo']] + n > a.repo_frac * a.token_budget:
            reason = 'repo_cap'
        elif self.source[skey] + n > self.source_cap_frac * a.token_budget:
            reason = 'source_cap'
        elif self.cls[m['resolved']] + n > a.class_frac * a.token_budget:
            reason = 'class_cap'
        else:
            self.inst[m['instance_id']] += 1
            self.repo[m['repo']] += n
            self.source[skey] += n
            self.cls[m['resolved']] += n
            self.total += n
            return True
        self.rejects[reason] += 1
        return False

def list_jobs(a) -> list[dict]:
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(REPO_ID, repo_type='dataset')
    jobs = [j for j in map(parse_source, files) if j]
    want_h = set(a.harnesses.split(','))
    jobs = [j for j in jobs if j['harness'] in want_h]
    if a.models:
        jobs = [j for j in jobs if j['model'] in set(a.models.split(','))]
    if a.sources:
        jobs = [j for j in jobs if j['source'] in set(a.sources.split(','))]
    jobs.sort(key=lambda j: j['path'])
    import random
    random.Random(a.seed).shuffle(jobs)
    return jobs[:a.max_files] if a.max_files else jobs

def _fmt(n: float) -> str:
    return f'{n / 1000000.0:.1f}M'

def run(a, tmp_dir: Path, env_vars: dict) -> dict:
    import psutil
    from ..env import probe_env, write_json
    t_start = time.time()
    tok_dir = ensure_tokenizer(a.tokenizer, tmp_dir / 'cache' / 'tok')
    tok_json = tok_dir / 'tokenizer.json'
    template = Path(a.hint_template).read_text() if a.hint_template else DEFAULT_TEMPLATE
    ncpu = os.cpu_count() or 1
    num_proc = a.num_proc or max(1, min(ncpu, 6))
    cfg = {'tokenizer_json': str(tok_json), 'max_seq_len': a.max_seq_len, 'overlength': a.overlength, 'hint_template': template, 'val_pct': a.val_pct, 'hf_cache': str(tmp_dir / 'cache' / 'hf' / 'hub'), 'rayon_threads': max(1, ncpu // num_proc), 'env': env_vars, 'resolved_values': [int(x) for x in a.resolved_values.split(',')]}
    jobs = list_jobs(a)
    print(f'[prep] {len(jobs)} parquet files, {num_proc} workers, budget {_fmt(a.token_budget)} tokens, max_seq_len {a.max_seq_len}, overlength={a.overlength}', flush=True)
    if a.inspect:
        return inspect(a, jobs, cfg)
    build_dir = tmp_dir / 'build' / 'recant_data'
    if build_dir.exists():
        import shutil
        shutil.rmtree(build_dir)
    writer = ShardWriter(build_dir, shard_tokens=a.shard_tokens, ram_frac=a.ram_frac, seed=a.seed)
    adm = Admission(a)
    agg: dict[str, Counter] = {}
    verify_result = None
    files_done = 0
    pending: dict = {}
    skipped: Counter = Counter()
    it = iter(enumerate(jobs))
    last_print = 0.0
    ctx = __import__('multiprocessing').get_context('spawn')
    ex = ProcessPoolExecutor(max_workers=num_proc, mp_context=ctx, initializer=_init_worker, initargs=(cfg,))
    zero_streak: Counter = Counter()
    has_rows: set = set()
    n_submitted = [0]
    all_sources = {source_key(j) for j in jobs}

    def dead(j: dict) -> bool:
        k = source_key(j)
        if zero_streak[k] >= a.dead_after and k not in has_rows:
            return True
        return adm.source[k] >= 0.98 * adm.source_cap_frac * a.token_budget

    def submit_next() -> bool:
        while True:
            try:
                i, j = next(it)
            except StopIteration:
                return False
            if dead(j):
                skipped[source_key(j)] += 1
                continue
            break
        j = dict(j, keep_raw=a.verify_template if n_submitted[0] < 2 * num_proc else 0)
        n_submitted[0] += 1
        pending[ex.submit(process_file, j)] = j
        return True
    try:
        for _ in range(num_proc + 1):
            submit_next()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                j = pending.pop(fut)
                try:
                    res = fut.result()
                except Exception as e:
                    print(f"[prep] job failed {j['path']}: {e!r}", flush=True)
                    agg.setdefault(source_key(j), Counter())['job_failed'] += 1
                    continue
                files_done += 1
                agg.setdefault(source_key(j), Counter()).update(res['stats'])
                if res['stats'].get('python_resolved01', 0):
                    has_rows.add(source_key(j))
                else:
                    zero_streak[source_key(j)] += 1
                adm.n_active = sum((1 for k in all_sources if not (zero_streak[k] >= a.dead_after and k not in has_rows)))
                if res['raw'] and verify_result is None and a.verify_template:
                    verify_result = _verify(res['raw'], tok_dir, tok_json)
                for rec in res['recs']:
                    if adm.done:
                        break
                    if adm.try_admit(rec):
                        writer.add(rec)
                if not adm.done:
                    submit_next()
            if adm.done:
                break
            now = time.time()
            if now - last_print > 10:
                last_print = now
                el = now - t_start
                print(f'[prep] files {files_done}/{len(jobs)}  tokens {_fmt(adm.total)}/{_fmt(a.token_budget)}  traj {sum(adm.inst.values())}  {adm.total / max(el, 1):,.0f} tok/s  ram {psutil.virtual_memory().percent:.0f}%  rejects {dict(adm.rejects)}', flush=True)
    finally:
        procs = list((getattr(ex, '_processes', None) or {}).values())
        ex.shutdown(wait=False, cancel_futures=True)
        for proc in procs:
            try:
                proc.terminate()
            except Exception:
                pass
    tok_hash = file_sha256(tok_json)
    manifest = writer.close({'tokenizer_sha256': tok_hash, 'tokenizer_source': a.tokenizer, 'turn_stride': TURN_STRIDE, 'max_seq_len': a.max_seq_len, 'overlength': a.overlength, 'vocab_size': None, 'special_ids': _special_ids(tok_json), 'hint_template': template, 'config': {k: v for k, v in vars(a).items() if not k.startswith('_')}})
    report = build_report(a, agg, adm, writer, manifest, verify_result, time.time() - t_start, probe_env())
    report['skipped_files_no_usable_rows'] = dict(skipped)
    write_json(build_dir / 'prep_report.json', report)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / 'prep_report.json', report)
    if not a.no_zip:
        z = zip_dir(build_dir, out_dir / 'recant_data.zip')
        print(f'[prep] wrote {z} ({z.stat().st_size / 2 ** 30:.2f} GB)', flush=True)
    print(f"[prep] done: {manifest['n_trajectories']} trajectories, {_fmt(manifest['total_tokens'])} tokens, {time.time() - t_start:.0f}s", flush=True)
    return report

def _special_ids(tok_json: Path) -> dict:
    from tokenizers import Tokenizer
    t = Tokenizer.from_file(str(tok_json))
    return {s: t.token_to_id(s) for s in ('<|im_start|>', '<|im_end|>', '<think>', '</think>', '<tool_call>', '</tool_call>', '<tool_response>', '</tool_response>', '<|endoftext|>')}

def _verify(raw: list[dict], tok_dir: Path, tok_json: Path) -> dict:
    try:
        from tokenizers import Tokenizer
        res = verify_against_hf(Renderer(Tokenizer.from_file(str(tok_json))), str(tok_dir), raw)
    except ImportError as e:
        res = {'skipped': f'transformers unavailable: {e}'}
    print(f'[prep] template verification: {res}', flush=True)
    return res

def zip_dir(src: Path, dst: Path) -> Path:
    with zipfile.ZipFile(dst, 'w', allowZip64=True) as z:
        for f in sorted(src.iterdir()):
            comp = zipfile.ZIP_STORED if f.suffix == '.bin' else zipfile.ZIP_DEFLATED
            z.write(f, f.name, compress_type=comp)
    return dst

def build_report(a, agg, adm, writer, manifest, verify_result, seconds, env) -> dict:
    total: Counter = Counter()
    per_source = {}
    unknown: Counter = Counter()
    for k, c in sorted(agg.items()):
        total.update(c)
        per_source[k] = {kk: v for kk, v in sorted(c.items()) if not kk.startswith('unknown_tool:')}
        seen = c.get('python_resolved01', 0)
        per_source[k]['pass_rate_after_filters'] = round(c.get('kept', 0) / seen, 3) if seen else None
        for kk, v in c.items():
            if kk.startswith('unknown_tool:'):
                unknown[kk.split(':', 1)[1]] += v
    lens = np.array([r['length'] for r in writer.rows]) if writer.rows else np.zeros(0)
    commits = np.array([r['n_commit'] for r in writer.rows]) if writer.rows else np.zeros(0)
    by_res = Counter((r['resolved'] for r in writer.rows))
    by_split = Counter((r['split'] for r in writer.rows))
    by_h = Counter((r['harness'] for r in writer.rows))
    return {'seconds': round(seconds, 1), 'manifest': manifest, 'template_verification': verify_result, 'admitted': {'trajectories': len(writer.rows), 'tokens': adm.total, 'tokens_by_resolved': dict(adm.cls), 'tokens_by_source': dict(adm.source), 'trajectories_by_resolved': dict(by_res), 'trajectories_by_split': dict(by_split), 'trajectories_by_harness': dict(by_h), 'rejects': dict(adm.rejects), 'length_pct': {p: int(np.percentile(lens, p)) for p in (10, 50, 90)} if len(lens) else None, 'commits_per_traj_pct': {p: int(np.percentile(commits, p)) for p in (10, 50, 90)} if len(commits) else None}, 'totals': {k: v for k, v in sorted(total.items()) if not k.startswith('unknown_tool:')}, 'per_source': per_source, 'unknown_tools_top20': unknown.most_common(20), 'env': env}

def inspect(a, jobs: list[dict], cfg: dict) -> dict:
    _init_worker(cfg)
    seen: dict[str, dict] = {}
    zeros: Counter = Counter()
    for j in jobs:
        k = source_key(j)
        if k in seen and seen[k]['python'] > 0 or zeros[k] >= a.dead_after:
            continue
        res = process_file(dict(j, max_rows=a.inspect_rows, keep_raw=a.verify_template or 20))
        seen[k] = {'python': res['stats'].get('python_resolved01', 0), 'res': res}
        if res['stats'].get('python_resolved01', 0):
            s = res['stats']
            print(f"[inspect] {k}: python_r01={s['python_resolved01']} kept={s.get('kept', 0)} drops={ {x: v for x, v in s.items() if x.startswith('drop_')}} unknown_tools={ {x.split(':', 1)[1]: v for x, v in s.items() if x.startswith('unknown_tool:')}} ({s['seconds']}s)", flush=True)
        else:
            zeros[k] += 1
            print(f"[inspect] {k}: no usable rows (python & resolved in {cfg['resolved_values']}) in {j['path']}", flush=True)
    raws = [r for v in seen.values() for r in v['res']['raw']][:a.verify_template or 20]
    tok_dir = Path(cfg['tokenizer_json']).parent
    out = {'verify': _verify(raws, tok_dir, Path(cfg['tokenizer_json'])) if raws else None}
    return out