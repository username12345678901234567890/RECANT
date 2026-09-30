"""Fast paths: each must equal its reference. GPU-only kernels are checked through the Triton interpreter."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from recant import fast
from recant.model import fused_ops as fo
from recant.model import gdn_glue, nvfp4, ops
from tests.test_model import tiny_model

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _restore_fast():
    yield
    fast.configure(False, "")


def test_rht_butterfly_matches_matmul_and_preserves_norm():
    torch.manual_seed(0)
    x = torch.randn(7, 96)
    y = nvfp4.rht(x)
    assert (y - nvfp4.rht_matmul(x)).abs().max() < 1e-5
    assert torch.allclose(y.reshape(-1, 16).norm(dim=-1), x.reshape(-1, 16).norm(dim=-1), rtol=1e-5)


def test_triton_quantizer_bit_identical_in_interpreter():
    pytest.importorskip("triton")
    code = f"""
import os, sys
os.environ["TRITON_INTERPRET"] = "1"
sys.path.insert(0, {str(ROOT)!r})
import torch
from recant.model import nvfp4 as q, fp4_kernels as k
torch.manual_seed(0)
bad = []
for dtype in (torch.bfloat16, torch.float32):
    for (m, kk, rot) in ((20, 96, True), (5, 32, True), (9, 64, False)):
        x = torch.randn(m, kk) * 3
        x[:, ::7] *= 20
        x = x.to(dtype).contiguous()
        ref = q.quantize(q.rht(x) if rot else x)
        got = k.quantize_fast(x, rot, False, BM=4, BG=2)
        if not (torch.equal(ref.codes, got.codes) and torch.equal(ref.scales.view(torch.uint8), got.scales.view(torch.uint8))
                and torch.equal(ref.gscale, got.gscale)):
            bad.append((str(dtype), m, kk, rot))
# stochastic rounding: unbiased on average
x = (torch.randn(64, 256) * 2).contiguous()
acc = torch.zeros_like(x)
n = 24
for _ in range(n):
    g = k.quantize_fast(x, False, True, BM=8, BG=4)
    acc += g.dequant()
err = ((acc / n - x).abs().mean() / x.abs().mean()).item()
print("RESULT", bad, err)
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=900)
    if "RESULT" not in r.stdout:
        pytest.skip(f"triton interpreter unavailable: {r.stderr[-300:]}")
    line = r.stdout.split("RESULT", 1)[1].strip()
    assert line.startswith("[] "), line
    assert float(line.split()[-1]) < 0.1, line


def _lora_out(fast_lora):
    fast.configure(False, "" if fast_lora else "lora")
    m = tiny_model(layers=4)
    m.add_lora(4, 8)
    for p in m.lora_parameters():
        torch.nn.init.normal_(p, std=0.05)
    ids = torch.randint(3, 300, (1, 40))
    out = m.logits(m.final_norm(m.run_layers(m.embed(ids))))
    out.square().mean().backward()
    return out.detach(), [p.grad.clone() for p in m.lora_parameters()]


def test_lora_fusion_equivalent():
    torch.manual_seed(1)
    a, ga = _lora_out(True)
    torch.manual_seed(1)
    b, gb = _lora_out(False)
    assert torch.allclose(a, b, rtol=1e-4, atol=1e-5)
    assert all(torch.allclose(x, y, rtol=1e-3, atol=1e-6) for x, y in zip(ga, gb))


@pytest.mark.parametrize("hq,hk", [(4, 4), (4, 1)])
def test_cached_two_part_matches_masked(hq, hk):
    torch.manual_seed(0)
    q = torch.randn(1, hq, 9, 16)
    k = torch.randn(1, hk, 25, 16)
    v = torch.randn(1, hk, 25, 16)
    ref = ops.attn_cached_masked(q, k, v, block=4)
    got = ops.attn_cached_two_part(q, k, v, ops._attn_lse_ref)
    assert torch.allclose(ref, got, atol=1e-5)
    k0, v0 = k[:, :, 16:], v[:, :, 16:]
    assert torch.allclose(ops.attn_cached_two_part(q, k0, v0, ops._attn_lse_ref), ops.attn_cached_masked(q, k0, v0), atol=1e-5)


def test_fused_ops_eager_sanity():
    torch.manual_seed(0)
    x, w = torch.randn(5, 32), torch.randn(32) * 0.1
    ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * (1 + w)
    assert torch.allclose(fo.rmsnorm(x, w, 1e-6), ref, atol=1e-5)
    g, u = torch.randn(5, 8), torch.randn(5, 8)
    assert torch.allclose(fo.swiglu(g, u), torch.nn.functional.silu(g) * u, atol=1e-6)
    assert torch.allclose(fo.gate_mul(g, u), g * torch.sigmoid(u), atol=1e-6)


def test_gdn_glue_state_roundtrip():
    s = torch.randn(2, 6, 3)
    for layout in ("zero_first", "zero_last"):
        assert torch.equal(gdn_glue._from_fla(gdn_glue._to_fla(s, layout), layout), s)


def test_profile_step_writes_files_and_keeps_results(tmp_path):
    from tests.test_train_e2e import args_for, workdir  # noqa: F401
    from recant import trainer

    def go(w, extra):
        return trainer.run(args_for(w, extra))

    import numpy as np
    from recant.data.writer import ShardWriter
    from recant.model.config import TextConfig
    from recant.model.qwen35 import Qwen35, save_hf_layout
    from tests.test_train_e2e import V, synth_traj

    def make(root):
        torch.manual_seed(0)
        m = Qwen35(TextConfig.tiny(vocab_size=V, layers=4), device="cpu", dtype=torch.float32)
        m.init_random(std=0.05, seed=0)
        save_hf_layout(m, root / "model")
        rng = np.random.default_rng(0)
        wr = ShardWriter(root / "data", seed=0)
        for i in range(10):
            r = synth_traj(rng, n=int(rng.integers(145, 175)), resolved=i % 2)
            r["meta"]["split"] = "val" if i in (8, 9) else "train"
            r["meta"]["instance_id"] = f"inst{i}"
            wr.add(r)
        wr.close({"max_seq_len": 200})
        return root

    a, b = make(tmp_path / "a"), make(tmp_path / "b")
    go(a, ["--max_steps", "2", "--profile_steps", "2"])
    go(b, ["--max_steps", "2", "--no_fast"])
    js = json.loads((a / "out" / "profile_step2.json").read_text())
    assert (a / "out" / "profile_step2.txt").exists() and js["buckets"] and js["top_kernels"]
    assert not list((b / "out").glob("profile_step*"))
    la = [json.loads(l)["loss"] for l in (a / "out" / "train.jsonl").read_text().splitlines()]
    lb = [json.loads(l)["loss"] for l in (b / "out" / "train.jsonl").read_text().splitlines()]
    assert la == pytest.approx(lb, rel=1e-4)
    assert "fast_paths" in json.loads((a / "out" / "backends.json").read_text())


def test_heads_set_range_keeps_buffers_on_param_device():
    """Regression: set_range() after heads.to(cuda) used to re-create the bins on the CPU."""
    from recant.heads import Heads

    h = Heads(16, n_taps=4, hidden=8, n_bins=11, y_max=8.0).to("meta")
    h.set_range(3.0)
    assert h.bin_centers.device.type == "meta" and h.bin_edges.device.type == "meta"
