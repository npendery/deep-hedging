"""Tests for deephedge/losses.py — convex risk-measure losses (spec §6, §13 tests 4–5, 7)."""

import math

import torch

from deephedge.losses import cvar_loss


def test_cvar_loss_matches_gaussian_closed_form_at_optimal_w():
    gen = torch.Generator().manual_seed(0)
    n = 400_000
    pnl = torch.randn(n, generator=gen)  # P&L ~ N(0,1) -> L = -pnl ~ N(0,1)
    alpha = 0.95

    # Closed-form Gaussian CVaR of the loss L: phi(z_alpha) / (1 - alpha).
    z = 1.6448536269514722  # standard-normal 0.95 quantile
    phi = math.exp(-z * z / 2.0) / math.sqrt(2.0 * math.pi)
    cvar_closed_form = phi / (1.0 - alpha)  # ~= 2.0627

    # Minimize F(w) over a grid bracketing VaR_0.95 (= z) using cvar_loss directly.
    ws = torch.linspace(1.0, 2.2, 1201)
    vals = torch.stack([cvar_loss(pnl, alpha, w) for w in ws])
    best_idx = int(torch.argmin(vals))
    best_w = float(ws[best_idx])
    best_cvar = float(vals[best_idx])

    assert abs(best_cvar - cvar_closed_form) < 0.05
    # Left endpoint of the argmin is VaR_alpha = z_0.95 (continuous P&L -> unique).
    assert abs(best_w - z) < 0.05

    # cvar_loss returns a scalar tensor.
    out = cvar_loss(pnl, alpha, torch.tensor(z))
    assert out.shape == torch.Size([])
