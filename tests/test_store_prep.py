import json
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("psutil")

from recant.data.prep import Admission, parse_source, split_of, truncate_rendered, zip_dir  # noqa: E402
from recant.data.render import T_CE_END, TURN_STRIDE  # noqa: E402
from recant.data.store import TrajectoryStore  # noqa: E402
from recant.data.writer import ShardWriter  # noqa: E402


def rec(i, n=100, resolved=0, repo="r", inst=None, harness="sweagent", model="m", source="s"):
    return {"ids": np.arange(n, dtype=np.uint32) + i * 1000, "hint": np.arange(4, dtype=np.uint32) + i,
            "turns": (np.arange(2 * TURN_STRIDE, dtype=np.int32).reshape(2, TURN_STRIDE) + i),
            "meta": {"traj_id": f"t{i}", "instance_id": inst or f"i{i}", "repo": repo, "harness": harness,
                     "model": model, "source": source, "resolved": resolved, "split": "train"}}


def test_writer_store_roundtrip(tmp_path):
    w = ShardWriter(tmp_path / "d", shard_tokens=800, seed=3)
    for i in range(25):
        w.add(rec(i, n=100 + i))
    m = w.close({"extra": 1})
    assert m["n_trajectories"] == 25 and m["n_shards"] >= 3 and m["extra"] == 1
    s = TrajectoryStore(tmp_path / "d")
    assert len(s) == 25
    for k in range(len(s)):
        r = s.get(k)
        i = int(r["traj_id"][1:])
        assert len(r["ids"]) == 100 + i and r["ids"][0] == i * 1000 and r["hint"][0] == i
        assert r["turns"].shape == (2, TURN_STRIDE) and r["turns"][0, 0] == i


def test_writer_flushes_under_ram_pressure(tmp_path, monkeypatch):
    import recant.data.writer as W

    monkeypatch.setattr(W, "_vm", lambda: SimpleNamespace(percent=99.0, available=1 << 20, total=1 << 30))
    w = ShardWriter(tmp_path / "d", seed=0)
    for i in range(5):
        w.add(rec(i))
    assert w.flushes == 5 and not w.buf  # every add flushed immediately


def test_zip_roundtrip(tmp_path):
    w = ShardWriter(tmp_path / "d", seed=0)
    w.add(rec(1))
    w.close()
    z = zip_dir(tmp_path / "d", tmp_path / "out.zip")
    names = zipfile.ZipFile(z).namelist()
    assert {"tokens_000.bin", "hints.bin", "index.parquet", "manifest.json"} <= set(names)


def args(**kw):
    base = dict(token_budget=1000, max_per_instance=2, repo_frac=0.5, source_frac=0.6, class_frac=0.7)
    base.update(kw)
    return SimpleNamespace(**base)


def test_admission_caps():
    a = Admission(args(source_frac=1.0, class_frac=1.0))  # isolate the instance and repo caps
    a.n_active = 10
    assert a.try_admit(rec(0, n=100, inst="x"))
    assert a.try_admit(rec(1, n=100, inst="x"))
    assert not a.try_admit(rec(2, n=100, inst="x")) and a.rejects["instance_cap"] == 1
    assert a.try_admit(rec(3, n=450, repo="big"))          # 450 <= repo_frac * budget (500)
    assert not a.try_admit(rec(4, n=100, repo="big"))      # 550 > 500
    assert a.rejects["repo_cap"] == 1 and a.total == 650


def test_admission_source_cap_relaxes_when_few_sources_alive():
    a = Admission(args(source_frac=0.35))
    a.n_active = 1
    assert a.source_cap_frac > 1.0  # a single alive source may fill the whole budget


def test_admission_budget_and_class_cap():
    a = Admission(args(class_frac=0.3, repo_frac=1.0, source_frac=1.0))
    a.n_active = 1
    assert a.try_admit(rec(0, n=250, resolved=1, repo="a", inst="a"))
    assert not a.try_admit(rec(1, n=250, resolved=1, repo="b", inst="b"))
    assert a.rejects["class_cap"] == 1
    assert a.try_admit(rec(2, n=250, resolved=0, repo="c", inst="c"))
    assert not a.done


def test_truncate_rendered_cuts_at_turn_boundary():
    ids = np.arange(100, dtype=np.uint32)
    turns = np.zeros((3, TURN_STRIDE), dtype=np.int32)
    turns[:, T_CE_END] = [30, 60, 95]
    out = truncate_rendered(ids, turns, 70)
    assert len(out[0]) == 60 and len(out[1]) == 2
    assert truncate_rendered(ids, turns, 10) is None


def test_parse_source_and_split():
    j = parse_source("data/openhands/minimax_m25/swe-rebench-v2/train-00001-of-00018.parquet")
    assert j == {"path": "data/openhands/minimax_m25/swe-rebench-v2/train-00001-of-00018.parquet",
                 "harness": "openhands", "model": "minimax_m25", "source": "swe-rebench-v2"}
    assert parse_source("README.md") is None
    assert split_of("a__b-1", 1) == split_of("a__b-1", 1)  # deterministic per instance
    assert sum(split_of(f"x{i}", 10) == "val" for i in range(2000)) == pytest.approx(200, abs=60)
