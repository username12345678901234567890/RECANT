"""Switches for the exact-result speed paths.

A path is ON only if it was requested (CLI) AND verified (startup self-check against the reference
implementation, see backends.select_fast_paths). Pure-torch paths whose equivalence is unit-tested on CPU
(`lora`) start verified; GPU-only paths (`quantize`, `elementwise`, `gdn_glue`, `attn`) start unverified.
"""
from __future__ import annotations

NAMES = ("quantize", "lora", "elementwise", "gdn_glue", "attn")
_requested = {n: True for n in NAMES}
_verified = {"quantize": False, "lora": True, "elementwise": False, "gdn_glue": False, "attn": False}


def configure(no_fast: bool = False, disable: str = "") -> None:
    for n in NAMES:
        _requested[n] = not no_fast
    for n in filter(None, (s.strip() for s in disable.split(","))):
        if n not in _requested:
            raise KeyError(f"unknown fast path {n!r}; have {NAMES}")
        _requested[n] = False


def set_verified(name: str, ok: bool) -> None:
    _verified[name] = bool(ok)


def enabled(name: str) -> bool:
    return _requested[name] and _verified[name]


def state() -> dict:
    return {n: {"requested": _requested[n], "verified": _verified[n], "on": enabled(n)} for n in NAMES}
