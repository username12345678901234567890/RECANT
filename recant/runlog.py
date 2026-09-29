"""Logging: stdout tee to a file, append-only JSONL metrics, system metrics."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


class _Tee:
    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, s):
        self.stream.write(s)
        self.fh.write(s)
        return len(s)

    def flush(self):
        self.stream.flush()
        self.fh.flush()

    def isatty(self):
        return False


class RunLog:
    """Everything goes to `out_dir` (Kaggle working). Files are tiny, so we write
    them directly and flush often; a crash still leaves the logs behind."""

    def __init__(self, out_dir: str | os.PathLike, tee_stdout: bool = True):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self._files: dict[str, object] = {}
        if tee_stdout:
            fh = open(self.dir / "stdout.log", "a", buffering=1)
            sys.stdout = _Tee(sys.__stdout__, fh)
            sys.stderr = _Tee(sys.__stderr__, fh)

    def jsonl(self, name: str, record: dict) -> None:
        fh = self._files.get(name)
        if fh is None:
            fh = self._files[name] = open(self.dir / f"{name}.jsonl", "a", buffering=1)
        record = {"t": round(time.time() - self.t0, 3), **record}
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def close(self) -> None:
        for fh in self._files.values():
            fh.close()
        self._files.clear()


def system_metrics() -> dict:
    """RAM/CPU (psutil) and GPU clock/power/temp/mem (nvml) if available."""
    m: dict = {}
    try:
        import psutil

        vm = psutil.virtual_memory()
        m.update(ram_used_gb=round((vm.total - vm.available) / 2**30, 2), ram_pct=vm.percent,
                 cpu_pct=psutil.cpu_percent(interval=None))
    except ImportError:
        pass
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        m.update(gpu_mem_used_gb=round(mem.used / 2**30, 2),
                 gpu_sm_mhz=pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
                 gpu_power_w=round(pynvml.nvmlDeviceGetPowerUsage(h) / 1000, 1),
                 gpu_temp_c=pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
    except Exception:  # noqa: BLE001
        pass
    return m
