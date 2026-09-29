"""Read-only memmap access to a prepared dataset directory."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .render import TURN_STRIDE


class TrajectoryStore:
    def __init__(self, data_dir: str | Path):
        import pyarrow.parquet as pq

        self.dir = Path(data_dir)
        self.manifest = json.loads((self.dir / "manifest.json").read_text())
        self.index = pq.read_table(self.dir / "index.parquet").to_pydict()
        self.n = self.manifest["n_trajectories"]
        self._shards: dict[int, np.memmap] = {}
        hints = self.dir / "hints.bin"
        self._hints = np.memmap(hints, dtype="<u4", mode="r") if hints.stat().st_size else np.zeros(0, "<u4")

    def _shard(self, i: int) -> np.memmap:
        if i not in self._shards:
            self._shards[i] = np.memmap(self.dir / f"tokens_{i:03d}.bin", dtype="<u4", mode="r")
        return self._shards[i]

    def __len__(self) -> int:
        return self.n

    def meta(self, i: int) -> dict:
        return {k: v[i] for k, v in self.index.items() if k != "turns"}

    def get(self, i: int) -> dict:
        ix = self.index
        off, n = ix["offset"][i], ix["length"][i]
        ho, hn = ix["hint_offset"][i], ix["hint_len"][i]
        return {
            "ids": self._shard(ix["shard"][i])[off:off + n],
            "hint": self._hints[ho:ho + hn],
            "turns": np.asarray(ix["turns"][i], dtype=np.int32).reshape(-1, TURN_STRIDE),
            **self.meta(i),
        }

    def split_indices(self, split: str) -> list[int]:
        return [i for i, s in enumerate(self.index["split"]) if s == split]
