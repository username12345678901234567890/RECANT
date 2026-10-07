from __future__ import annotations
import math
import time

class TimeBudget:

    def __init__(self, limit_hours: float, reserve_min: float, t0: float | None=None):
        self.t0 = time.time() if t0 is None else t0
        self.deadline = self.t0 + limit_hours * 3600 - reserve_min * 60
        self.hard_deadline = self.t0 + limit_hours * 3600
        self.train_start: float | None = None

    def start_training(self):
        self.train_start = time.time()

    def elapsed(self) -> float:
        return time.time() - self.t0

    def frac(self) -> float:
        if self.train_start is None:
            return 0.0
        span = max(self.deadline - self.train_start, 1e-06)
        return min(1.0, (time.time() - self.train_start) / span)

    def done(self) -> bool:
        return time.time() >= self.deadline

    def remaining(self) -> float:
        return self.deadline - time.time()

def lr_at(step: int, frac: float, base: float, warmup_steps: int=20, min_ratio: float=0.1) -> float:
    warm = min(1.0, (step + 1) / max(warmup_steps, 1))
    cos = min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(frac, 1.0)))
    return base * warm * cos

class HeadGain:

    def __init__(self, open_steps: int, open_frac: float, ramp_steps: int, gain_max: float):
        self.open_steps, self.open_frac, self.ramp_steps, self.gain_max = (open_steps, open_frac, ramp_steps, gain_max)
        self.opened_at: int | None = None

    def __call__(self, step: int, frac: float) -> float:
        if self.opened_at is None and (step >= self.open_steps or frac >= self.open_frac):
            self.opened_at = step
        if self.opened_at is None:
            return 0.0
        return self.gain_max * min(1.0, (step - self.opened_at + 1) / max(self.ramp_steps, 1))