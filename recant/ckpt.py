from __future__ import annotations
import json
import time
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file

def _cpu(sd: dict) -> dict:
    return {k: v.detach().to('cpu').contiguous() for k, v in sd.items()}

def save_final(out_dir, lora_sd: dict, heads_sd: dict, meta: dict, log=print) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file(_cpu(lora_sd), str(out / 'adapter.safetensors'))
    save_file(_cpu(heads_sd), str(out / 'heads.safetensors'))
    (out / 'config.json').write_text(json.dumps(meta.get('config', {}), indent=2, default=str))
    (out / 'trainer_state.json').write_text(json.dumps(meta.get('state', {}), indent=2, default=str))
    sizes = {p.name: round(p.stat().st_size / 2 ** 20, 1) for p in out.iterdir() if p.suffix == '.safetensors'}
    log(f'[ckpt] saved final adapter/heads to {out}: {sizes} MB')
    return sizes

def load_final(path, model, heads):
    p = Path(path)
    model.load_lora_state_dict(load_file(str(p / 'adapter.safetensors')))
    heads.load_state_dict(load_file(str(p / 'heads.safetensors')))

class Shadow:

    def __init__(self):
        self.lora: dict | None = None
        self.heads: dict | None = None
        self.state: dict = {}
        self.at = 0.0

    def update(self, model, heads, state: dict | None=None):
        self.lora = _cpu(model.lora_state_dict())
        self.heads = _cpu(heads.state_dict())
        self.state = dict(state or {})
        self.at = time.time()

    def save(self, out_dir, meta: dict, log=print):
        if self.lora is None:
            raise RuntimeError('no shadow copy yet')
        meta = {**meta, 'state': {**meta.get('state', {}), **self.state, 'from_shadow': True}}
        return save_final(out_dir, self.lora, self.heads, meta, log)

def smoke_test_save(tmp_dir, model, heads, log=print):
    d = Path(tmp_dir) / 'save_smoke'
    save_final(d, model.lora_state_dict(), heads.state_dict(), {'config': {'smoke': True}}, log=lambda *_: None)
    a = load_file(str(d / 'adapter.safetensors'))
    ok = set(a) == set(model.lora_state_dict()) and all((torch.equal(a[k], v.detach().cpu()) for k, v in model.lora_state_dict().items()))
    h = load_file(str(d / 'heads.safetensors'))
    ok = ok and set(h) == set(heads.state_dict())
    for f in d.iterdir():
        f.unlink()
    d.rmdir()
    log(f"[ckpt] save/load smoke test: {('ok' if ok else 'FAILED')}")
    if not ok:
        raise RuntimeError('save/load smoke test failed')
    return ok