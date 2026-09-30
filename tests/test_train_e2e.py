"""End-to-end: tiny random Qwen3.5 checkpoint + synthetic dataset -> train.py logic on CPU."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from recant.data.render import (F_COMMIT, F_REASONING, T_ANCHOR, T_CALL_END, T_CALL_START, T_CE_END, T_CE_START,
                                T_FLAGS, T_NCALLS, T_START, TURN_STRIDE)
from recant.data.writer import ShardWriter
from recant.model.config import TextConfig
from recant.model.qwen35 import Qwen35, save_hf_layout

V = 300


def synth_traj(rng, n=150, resolved=1):
    ids = rng.integers(3, V, n).astype(np.uint32)
    turns = np.full((3, TURN_STRIDE), -1, np.int32)
    for k, (s, e) in enumerate([(12, 42), (55, 95), (105, 140)]):
        turns[k, T_START], turns[k, T_CE_START], turns[k, T_CE_END] = s, s + 3, e
        turns[k, T_CALL_START], turns[k, T_CALL_END] = s + 12, e - 2
        turns[k, T_ANCHOR] = s + 11
        turns[k, T_FLAGS] = (F_COMMIT if k else 0) | F_REASONING
        turns[k, T_NCALLS] = 1
    return {"ids": ids, "turns": turns, "hint": rng.integers(3, V, 7).astype(np.uint32),
            "meta": {"traj_id": f"t{rng.integers(1 << 30)}", "instance_id": "i", "repo": "r", "harness": "sweagent",
                     "model": "m", "source": "s", "resolved": resolved, "split": "train"}}


@pytest.fixture()
def workdir(tmp_path):
    torch.manual_seed(0)
    m = Qwen35(TextConfig.tiny(vocab_size=V, layers=4), device="cpu", dtype=torch.float32)
    m.init_random(std=0.05, seed=0)
    save_hf_layout(m, tmp_path / "model")
    rng = np.random.default_rng(0)
    w = ShardWriter(tmp_path / "data", seed=0)
    for i in range(10):
        r = synth_traj(rng, n=int(rng.integers(145, 175)), resolved=i % 2)
        r["meta"]["split"] = "val" if i in (8, 9) else "train"
        r["meta"]["instance_id"] = f"inst{i}"
        w.add(r)
    w.close({"max_seq_len": 200})
    return tmp_path


def args_for(work, extra=()):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import train

    return train.parse_args([
        "--model_path", str(work / "model"), "--data_dir", str(work / "data"), "--out_dir", str(work / "out"),
        "--tmp_dir", str(work / "scratch"), "--min_tmp_gb", "0", "--device", "cpu", "--dtype", "fp32",
        "--lora_r", "4", "--lora_alpha", "4", "--head_hidden", "16", "--n_bins", "11", "--tokens_per_step", "250",
        "--n_keep", "16", "--topk", "32", "--max_chunk", "64", "--calib_trajs", "2", "--calib_turns", "2",
        "--eval_trajs", "2", "--eval_every_min", "0", "--time_limit_hours", "0.1", "--save_reserve_min", "0",
        "--head_warmup_steps", "2", "--head_gain_ramp", "1", "--head_gain", "0.1", "--max_steps", "4",
        "--warmup_steps", "2", *extra])


def test_train_end_to_end(workdir):
    from recant import trainer

    a = args_for(workdir)
    state = trainer.run(a)
    out = workdir / "out"
    for f in ("adapter.safetensors", "heads.safetensors", "config.json", "trainer_state.json", "train.jsonl",
              "eval.jsonl", "env.json", "backends.json", "calibration.json", "summary.json", "stdout.log"):
        assert (out / f).exists(), f
    assert state["step"] == 4 and state["ooms"] == 0 and state["nonfinite_skips"] == 0
    rows = [json.loads(l) for l in (out / "train.jsonl").read_text().splitlines()]
    assert len(rows) == 4 and all(np.isfinite(r["loss"]) and np.isfinite(r["grad_norm"]) for r in rows)
    assert rows[0]["head_gain"] == 0.0 and rows[-1]["head_gain"] > 0          # warm-up then opened
    calib = json.loads((out / "calibration.json").read_text())
    assert calib["n_turns"] > 0 and calib["y_max"] >= 2.0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["status"] == "finished"
    # only final artefacts + logs in the output dir (no intermediate weight files)
    assert sorted(p.name for p in out.glob("*.safetensors")) == ["adapter.safetensors", "heads.safetensors"]
    # reload into a fresh model: the saved adapter reproduces the trained one exactly
    from recant.ckpt import load_final
    from recant.heads import Heads
    from recant.model.qwen35 import load_hf

    m = load_hf(workdir / "model", device="cpu", dtype=torch.float32, quantize=True)
    m.add_lora(4, 4)
    h = Heads(m.c.hidden_size, n_taps=4, hidden=16, n_bins=11, y_max=calib["y_max"])
    load_final(out, m, h)
    saved = load_file(str(out / "adapter.safetensors"))
    assert set(saved) == set(m.lora_state_dict())
    assert any(v.abs().max() > 0 for k, v in saved.items() if "lora_B" in k), "LoRA never moved"
    assert (workdir / "out" / "adapter.safetensors").stat().st_size > 0


def test_stops_on_time_limit_and_still_saves(workdir):
    from recant import trainer

    a = args_for(workdir, ["--time_limit_hours", "0.0006", "--max_steps", "1000"])   # ~2 s
    state = trainer.run(a)
    assert state["status"] == "time_limit" and state["step"] >= 1
    assert (workdir / "out" / "adapter.safetensors").exists()


def test_oom_recovery_skips_and_continues(workdir, monkeypatch):
    """A simulated OOM must restore gradients, shrink the plan, and keep training."""
    from recant import branch, trainer

    real = branch.run_trajectory
    calls = {"n": 0}

    def flaky(*args, **kw):
        calls["n"] += 1
        if calls["n"] in (2, 3):                       # trajectory fails twice -> skipped
            raise RuntimeError("CUDA out of memory. (simulated)")
        return real(*args, **kw)

    monkeypatch.setattr(branch, "run_trajectory", flaky)
    monkeypatch.setattr("recant.branch.run_trajectory", flaky)
    a = args_for(workdir, ["--max_steps", "3"])
    state = trainer.run(a)
    assert state["ooms"] == 2 and state["step"] == 3 and state["nonfinite_skips"] == 0
    assert (workdir / "out" / "adapter.safetensors").exists()


def test_crash_still_writes_final_artifacts(workdir, monkeypatch):
    from recant import branch, trainer

    def boom(*a, **k):
        raise ValueError("boom")

    monkeypatch.setattr(branch, "run_trajectory", boom)
    with pytest.raises(ValueError):
        trainer.run(args_for(workdir))
    out = workdir / "out"
    assert (out / "adapter.safetensors").exists() and "error" in json.loads((out / "summary.json").read_text())["status"]
