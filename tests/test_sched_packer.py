import time

import numpy as np

from recant.data.packer import MemoryModel, StepPlanner
from recant.sched import HeadGain, TimeBudget, lr_at

GB = 2**30


def test_lr_warmup_then_time_driven_cosine():
    assert lr_at(0, 0.0, 1.0, warmup_steps=10) == 0.1
    assert lr_at(50, 0.0, 1.0, warmup_steps=10) == 1.0
    mid, end = lr_at(50, 0.5, 1.0), lr_at(50, 1.0, 1.0, min_ratio=0.1)
    assert 0.1 < mid < 1.0 and abs(end - 0.1) < 1e-9
    assert lr_at(50, 0.3, 1.0) > lr_at(50, 0.7, 1.0)          # decays with the TIME fraction, not the step


def test_head_gain_opens_by_steps_or_time_then_ramps():
    g = HeadGain(open_steps=100, open_frac=0.5, ramp_steps=10, gain_max=0.1)
    assert g(0, 0.0) == 0.0 and g(99, 0.4) == 0.0
    assert abs(g(100, 0.4) - 0.1 / 10) < 1e-12
    assert abs(g(109, 0.4) - 0.1) < 1e-9 and g(500, 0.9) == 0.1
    h = HeadGain(open_steps=10**9, open_frac=0.2, ramp_steps=1, gain_max=0.1)
    assert h(3, 0.1) == 0.0 and h(4, 0.25) == 0.1               # time fraction opens it early


def test_time_budget_reserve_and_fraction():
    b = TimeBudget(limit_hours=1.0, reserve_min=12.0, t0=time.time() - 10)
    assert not b.done() and b.frac() == 0.0
    b.start_training()
    assert 0.0 <= b.frac() < 0.01
    late = TimeBudget(limit_hours=0.001, reserve_min=0.0, t0=time.time() - 100)
    assert late.done()
    assert abs((b.hard_deadline - b.deadline) - 12 * 60) < 1e-6


def test_memory_model_fits_slope_and_reanchors():
    m = MemoryModel(static_bytes=20 * GB, bytes_per_token=0.6 * 2**20)
    for t in (20000, 40000, 60000, 80000, 30000):
        m.observe(t, 25 * GB + 0.9 * 2**20 * t)               # true slope 0.9 MB/token
    assert abs(m.c1 / 2**20 - 0.9) < 0.02 and abs(m.c0 / GB - 25) < 0.5
    assert m.predict(98304) > MemoryModel(20 * GB).predict(98304)


def test_planner_accumulates_skips_oversized_and_backs_off_on_oom():
    lengths = [30000, 60000, 90000, 50000, 70000, 40000]
    mem = MemoryModel(static_bytes=20 * GB, bytes_per_token=0.6 * 2**20)
    p = StepPlanner(list(range(6)), lengths, tokens_per_step=100_000, mem=mem, budget_bytes=70 * GB,
                    max_len=98304, seed=0)
    for _ in range(4):
        step = p.next_step()
        assert step and sum(lengths[i] for i in step) >= 100_000
        assert all(mem.predict(lengths[i]) <= 70 * GB for i in step)
    tight = StepPlanner(list(range(6)), lengths, 100_000, MemoryModel(20 * GB), budget_bytes=45 * GB, max_len=98304)
    step = tight.next_step()
    assert all(lengths[i] <= 45000 for i in step) and tight.skipped_memory >= 0     # (45-20)GB/0.6MB ~ 42.6K tokens
    cap0 = p.token_cap
    p.report_oom(60000)
    assert p.token_cap == 54000 and p.ooms == 1 and p.token_cap < cap0
    for _ in range(50):
        p.end_step_clean()
    assert p.token_cap > 54000                                  # slowly recovers after clean steps


def test_planner_reports_exhaustion_when_nothing_fits():
    p = StepPlanner([0, 1], [90000, 95000], 100_000, MemoryModel(20 * GB), budget_bytes=30 * GB, max_len=98304)
    assert p.next_step() == [] and p.exhausted
