"""RECANT heads: multi-layer mix -> correction head (delta), magnitude head (m_hat), gate g.

logits = W_U (h_L + g * delta)   with   g = sigmoid(a * E[m_hat] + b),   E[m_hat] stop-grad.
Heads are kept in fp32 (tiny; avoids bf16 optimizer issues).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradScale(torch.autograd.Function):
    """Identity in forward; multiplies the gradient by `gain` in backward."""

    @staticmethod
    def forward(ctx, x, gain):
        ctx.gain = gain
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g * ctx.gain, None


class Heads(nn.Module):
    def __init__(self, d: int, n_taps: int = 4, hidden: int = 1024, n_bins: int = 51, y_max: float = 8.0,
                 gate_bias: float = -4.0):
        super().__init__()
        self.n_taps, self.n_bins = n_taps, n_bins
        self.tap_norms = nn.ModuleList(nn.LayerNorm(d, elementwise_affine=False) for _ in range(n_taps))
        self.mix_logits = nn.Parameter(torch.zeros(n_taps))
        self.ln_delta, self.ln_mag = nn.LayerNorm(d), nn.LayerNorm(d)
        self.delta_mlp = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))
        self.mag_mlp = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, n_bins))
        nn.init.zeros_(self.delta_mlp[2].weight)
        nn.init.zeros_(self.delta_mlp[2].bias)
        self.gate_a = nn.Parameter(torch.tensor(1.0))
        self.gate_b = nn.Parameter(torch.tensor(gate_bias))
        self.gate_bias0 = gate_bias
        self.set_range(y_max, reset_gate=True)

    def set_range(self, y_max: float, reset_gate: bool = False):
        """log1p(KL) range covered by the magnitude bins (fixed after calibration).
        With reset_gate, b is re-centered so the gate starts at sigmoid(gate_bias0) ~ 0 even though a
        uniform m_hat has E = y_max/2 (call before training only)."""
        if reset_gate:
            with torch.no_grad():
                self.gate_b.fill_(self.gate_bias0 - float(self.gate_a) * float(y_max) / 2)
        edges = torch.linspace(0.0, float(y_max), self.n_bins + 1)
        self.register_buffer("bin_edges", edges, persistent=True)
        self.register_buffer("bin_centers", (edges[:-1] + edges[1:]) / 2, persistent=True)
        self.y_max = float(y_max)

    def mix(self, taps: list[torch.Tensor]) -> torch.Tensor:
        w = torch.softmax(self.mix_logits, 0)
        return sum(w[i] * self.tap_norms[i](t.float()) for i, t in enumerate(taps))

    def delta(self, h_mix):
        return self.delta_mlp(self.ln_delta(h_mix))

    def mag_logits(self, h_mix):
        return self.mag_mlp(self.ln_mag(h_mix))

    def expected_mag(self, logits):
        return (torch.softmax(logits, -1) * self.bin_centers).sum(-1)

    def gate(self, mag_logits):
        """g = sigmoid(a * sg(E[m_hat]) + b)  (design invariant 4: m_hat is stop-grad here)."""
        return torch.sigmoid(self.gate_a * self.expected_mag(mag_logits.detach()) + self.gate_b)

    def hl_gauss_loss(self, mag_logits, y):
        """Cross-entropy against a truncated Gaussian over the bins centered at y (sigma = 0.75 bin width)."""
        width = (self.bin_edges[1] - self.bin_edges[0]).item()
        sigma = 0.75 * width
        y = y.float().clamp(0.0, self.y_max - 1e-6)[:, None]
        cdf = 0.5 * (1 + torch.erf((self.bin_edges[None, :] - y) / (sigma * math.sqrt(2))))
        p = (cdf[:, 1:] - cdf[:, :-1])
        p = p / p.sum(-1, keepdim=True).clamp(min=1e-12)
        return -(p * F.log_softmax(mag_logits.float(), -1)).sum(-1)   # per-anchor loss
