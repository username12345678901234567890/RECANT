from __future__ import annotations
import json
from pathlib import Path
from . import prof

class StepProfiler:

    def __init__(self, out_dir, step: int, log=print):
        self.out, self.step, self.log, self.p = (Path(out_dir), step, log, None)

    def __enter__(self):
        import torch
        acts = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            acts.append(torch.profiler.ProfilerActivity.CUDA)
        self.p = torch.profiler.profile(activities=acts)
        self.p.__enter__()
        prof.set_active(True)
        return self

    def __exit__(self, *exc):
        prof.set_active(False)
        try:
            self.p.__exit__(*exc)
            self._write()
        except Exception as e:
            self.log(f'[profile] failed: {e!r}')
        return False

    def _write(self):
        import torch
        cuda = torch.cuda.is_available()
        evs = self.p.key_averages()
        key = 'self_device_time_total' if cuda else 'self_cpu_time_total'
        tkey = 'device_time_total' if cuda else 'cpu_time_total'

        def t(e, k, alt):
            return float(getattr(e, k, None) if getattr(e, k, None) is not None else getattr(e, alt, 0.0))
        buckets, phases = ({}, {})
        for e in evs:
            if e.key.startswith('b:'):
                buckets[e.key[2:]] = t(e, tkey, 'cuda_time_total') / 1000000.0
            elif e.key.startswith('p:'):
                phases[e.key[2:]] = t(e, tkey, 'cuda_time_total') / 1000000.0
        top = sorted(evs, key=lambda e: -t(e, key, 'self_cuda_time_total'))[:30]
        rows = [{'name': e.key[:120], 'calls': e.count, 'self_s': t(e, key, 'self_cuda_time_total') / 1000000.0} for e in top]
        rep = {'step': self.step, 'unit': 's', 'device': 'cuda' if cuda else 'cpu', 'buckets': buckets, 'phases': phases, 'top_kernels': rows}
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / f'profile_step{self.step}.json').write_text(json.dumps(rep, indent=1))
        lines = [f"profile step {self.step} ({rep['device']})", 'buckets (s): ' + ', '.join((f'{k}={v:.3f}' for k, v in sorted(buckets.items()))), 'phases  (s): ' + ', '.join((f'{k}={v:.3f}' for k, v in sorted(phases.items()))), 'top kernels:']
        lines += [f"  {r['self_s']:9.4f}s x{r['calls']:<6} {r['name']}" for r in rows]
        (self.out / f'profile_step{self.step}.txt').write_text('\n'.join(lines) + '\n')
        self.log('[profile] ' + lines[1])