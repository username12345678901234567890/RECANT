"""VRAM-aware step planning.

Trajectories are 40-100K tokens, so a micro-batch is one trajectory (each already fills the GPU).
What varies is memory: peak ~ c0 + c1 * tokens. The planner
  * fits (c0, c1) online from observed peaks (with a prior from the architecture estimate),
  * skips trajectories that would not fit the current budget instead of crashing into an OOM,
  * on a real OOM lowers the per-trajectory token cap (AIMD: x0.9 on OOM, x1.02 after clean steps),
  * accumulates trajectories until `tokens_per_step` is reached (one optimizer step).
"""
from __future__ import annotations

import numpy as np


class MemoryModel:
    """peak_bytes ~ c0 + c1 * tokens."""

    def __init__(self, static_bytes: float, bytes_per_token: float = 0.6 * 2**20, keep: int = 64):
        self.c0, self.c1 = float(static_bytes), float(bytes_per_token)
        self.obs: list[tuple[int, float]] = []
        self.keep = keep

    def predict(self, tokens: int) -> float:
        return self.c0 + self.c1 * tokens

    def observe(self, tokens: int, peak_bytes: float):
        self.obs = (self.obs + [(tokens, float(peak_bytes))])[-self.keep:]
        if len(self.obs) >= 4:
            t = np.array([o[0] for o in self.obs], float)
            p = np.array([o[1] for o in self.obs], float)
            if t.max() - t.min() > 0.1 * t.max():          # enough spread to fit a slope
                c1, c0 = np.polyfit(t, p, 1)
                if c1 > 0:
                    self.c1, self.c0 = float(c1), float(max(c0, 0.0))
                    return
        # otherwise: keep the slope, re-anchor the intercept on the worst observation
        self.c0 = max(self.c0, float(peak_bytes) - self.c1 * tokens)


class StepPlanner:
    def __init__(self, indices: list[int], lengths: list[int], tokens_per_step: int, mem: MemoryModel,
                 budget_bytes: float, max_len: int, seed: int = 0):
        self.idx = np.array(indices)
        self.len = {int(i): int(lengths[i]) for i in indices}
        self.tps = tokens_per_step
        self.mem = mem
        self.budget = float(budget_bytes)
        self.token_cap = float(max_len)
        self.max_len = max_len
        self.rng = np.random.default_rng(seed)
        self.order: list[int] = []
        self.epoch = -1
        self.skipped_memory = 0
        self.ooms = 0
        self.clean_steps = 0
        self.exhausted = False

    def _refill(self):
        self.epoch += 1
        self.order = self.rng.permutation(self.idx).tolist()

    def fits(self, tokens: int) -> bool:
        return tokens <= self.token_cap and self.mem.predict(tokens) <= self.budget

    def next_step(self) -> list[int]:
        step, tot, scanned = [], 0, 0
        while tot < self.tps and scanned <= 2 * len(self.idx) + 8:
            if not self.order:
                self._refill()
            i = self.order.pop()
            scanned += 1
            n = self.len[i]
            if not self.fits(n):
                self.skipped_memory += 1
                continue
            step.append(i)
            tot += n
        if not step:
            self.exhausted = True
        return step

    def report_ok(self, tokens: int, peak_bytes: float | None):
        if peak_bytes:
            self.mem.observe(tokens, peak_bytes)

    def end_step_clean(self):
        self.clean_steps += 1
        if self.clean_steps >= 5 and self.token_cap < self.max_len:
            self.token_cap = min(float(self.max_len), self.token_cap * 1.02)

    def report_oom(self, tokens: int):
        self.ooms += 1
        self.clean_steps = 0
        self.token_cap = max(4096.0, min(self.token_cap, tokens) * 0.9)
