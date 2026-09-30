"""Profiler labels. Labels are only created while a profile step is running (zero cost otherwise).

Leaf buckets ("b:*") never nest, so their inclusive device time can be summed per bucket."""
from __future__ import annotations

from contextlib import nullcontext

ACTIVE = False


def rf(name: str):
    if not ACTIVE:
        return _NULL
    import torch

    return torch.profiler.record_function(name)


_NULL = nullcontext()


def set_active(flag: bool) -> None:
    global ACTIVE
    ACTIVE = bool(flag)
