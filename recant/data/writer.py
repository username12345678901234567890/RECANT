from __future__ import annotations
import json
import random
from pathlib import Path
import numpy as np
FORMAT_VERSION = 1

def _vm():
    import psutil
    return psutil.virtual_memory()

class ShardWriter:

    def __init__(self, out_dir: str | Path, shard_tokens: int=1 << 27, ram_frac: float=0.7, seed: int=0, min_buffer_bytes: int=64 << 20):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.shard_tokens = shard_tokens
        self.ram_frac = ram_frac
        self.min_buffer_bytes = min_buffer_bytes
        self.rng = random.Random(seed)
        self.buf: list[dict] = []
        self.buf_bytes = 0
        self.rows: list[dict] = []
        self.shard_id = -1
        self.shard_fill = 0
        self._shard_fh = None
        self._hint_fh = open(self.dir / 'hints.bin', 'wb')
        self.hint_pos = 0
        self.total_tokens = 0
        self.flushes = 0
        self.max_buffer_bytes_seen = 0

    def _cap_bytes(self) -> int:
        vm = _vm()
        headroom = vm.available - (1.0 - self.ram_frac) * vm.total
        return int(max(self.min_buffer_bytes, min(self.shard_tokens * 4, 0.3 * max(headroom, 0))))

    def add(self, rec: dict) -> None:
        self.buf.append(rec)
        self.buf_bytes += rec['ids'].nbytes + rec['hint'].nbytes + rec['turns'].nbytes
        self.max_buffer_bytes_seen = max(self.max_buffer_bytes_seen, self.buf_bytes)
        if self.buf_bytes >= self._cap_bytes() or _vm().percent >= self.ram_frac * 100:
            self.flush()

    def _open_next_shard(self) -> None:
        if self._shard_fh:
            self._shard_fh.close()
        self.shard_id += 1
        self.shard_fill = 0
        self._shard_fh = open(self.dir / f'tokens_{self.shard_id:03d}.bin', 'wb')

    def flush(self) -> None:
        if not self.buf:
            return
        self.rng.shuffle(self.buf)
        for rec in self.buf:
            n = len(rec['ids'])
            if self._shard_fh is None or self.shard_fill + n > self.shard_tokens:
                self._open_next_shard()
            self._shard_fh.write(rec['ids'].astype('<u4', copy=False).tobytes())
            hint = rec['hint']
            self._hint_fh.write(hint.astype('<u4', copy=False).tobytes())
            row = dict(rec['meta'])
            row.update(shard=self.shard_id, offset=self.shard_fill, length=n, hint_offset=self.hint_pos, hint_len=len(hint), turns=rec['turns'].reshape(-1).astype(np.int32).tolist())
            self.rows.append(row)
            self.shard_fill += n
            self.hint_pos += len(hint)
            self.total_tokens += n
        self.buf.clear()
        self.buf_bytes = 0
        self.flushes += 1

    def close(self, manifest_extra: dict | None=None) -> dict:
        import pyarrow as pa
        import pyarrow.parquet as pq
        self.flush()
        if self._shard_fh:
            self._shard_fh.close()
        self._hint_fh.close()
        self.rng.shuffle(self.rows)
        cols: dict[str, list] = {}
        for r in self.rows:
            for k, v in r.items():
                cols.setdefault(k, []).append(v)
        table = pa.table({k: pa.array(v) for k, v in cols.items()}) if cols else pa.table({})
        pq.write_table(table, self.dir / 'index.parquet', compression='zstd')
        manifest = {'format_version': FORMAT_VERSION, 'n_trajectories': len(self.rows), 'total_tokens': self.total_tokens, 'n_shards': self.shard_id + 1, 'shard_tokens': self.shard_tokens, 'hint_tokens': self.hint_pos, 'flushes': self.flushes, 'max_buffer_mb': round(self.max_buffer_bytes_seen / 2 ** 20, 1), **(manifest_extra or {})}
        (self.dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        return manifest