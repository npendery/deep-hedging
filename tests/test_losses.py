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


def test_cvar_geq_var_geq_expected_loss():
    gen = torch.Generator().manual_seed(1)
    n = 200_000
    # Skewed, profit-positive P&L: mostly small positive, with a fat left tail (losses).
    pnl = 0.5 - torch.relu(torch.randn(n, generator=gen)) ** 2
    alpha = 0.95
    loss = -pnl

    var = torch.quantile(loss, alpha)            # VaR_alpha of the loss
    expected_loss = loss.mean()                  # E[L]
    cvar = cvar_loss(pnl, alpha, var)            # CVaR_alpha = F(w=VaR)

    assert float(cvar) >= float(var) - 1e-4
    assert float(var) >= float(expected_loss) - 1e-4
    # Tail is genuinely heavier than the mean: strict gaps on this skewed sample.
    assert float(cvar) > float(var)
    assert float(var) > float(expected_loss)


from deephedge.losses import entropic_loss


def test_entropic_small_lambda_limit_recovers_negative_mean_pnl():
    gen = torch.Generator().manual_seed(2)
    n = 200_000
    pnl = (0.3 + 1.5 * torch.randn(n, generator=gen)).to(torch.float64)

    neg_mean = -pnl.mean()
    val = entropic_loss(pnl, lam=1e-4)

    assert val.shape == torch.Size([])
    assert abs(float(val) - float(neg_mean)) < 1e-3


def test_entropic_equals_mean_form_identity():
    gen = torch.Generator().manual_seed(3)
    n = 50_000
    pnl = (0.2 * torch.randn(n, generator=gen)).to(torch.float64)
    lam = 1.0

    impl = entropic_loss(pnl, lam)
    mean_form = (1.0 / lam) * torch.log(torch.mean(torch.exp(-lam * pnl)))

    assert abs(float(impl) - float(mean_form)) < 1e-5

    # The sum form (no -log N) must differ by exactly (1/lam)*log(N).
    sum_form = (1.0 / lam) * torch.logsumexp(-lam * pnl, dim=0)
    assert abs(float(sum_form - impl) - (1.0 / lam) * math.log(n)) < 1e-6


def test_losses_are_differentiable_through_pnl_parameter():
    gen = torch.Generator().manual_seed(4)
    n = 20_000
    base = torch.randn(n, generator=gen)

    # CVaR: pnl = theta * base ; w is a separate learnable scalar.
    theta_cvar = torch.nn.Parameter(torch.tensor(1.0))
    w = torch.nn.Parameter(torch.tensor(1.6))
    pnl_cvar = theta_cvar * base
    loss_cvar = cvar_loss(pnl_cvar, 0.95, w)
    loss_cvar.backward()
    assert theta_cvar.grad is not None and torch.isfinite(theta_cvar.grad)
    assert w.grad is not None and torch.isfinite(w.grad)
    # d/dw F = 1 - (1/(1-alpha)) * P(L > w). At w below the 0.95 loss-quantile the tail
    # mass exceeds (1-alpha), so the gradient w.r.t. w is negative (lower w not optimal).
    assert float(w.grad) < 0.0

    # Entropic: pnl = theta * base + shift ; gradient w.r.t. an additive shift is -1.
    theta_ent = torch.nn.Parameter(torch.tensor(0.5))
    shift = torch.nn.Parameter(torch.tensor(0.0))
    pnl_ent = theta_ent * base + shift
    loss_ent = entropic_loss(pnl_ent, lam=1.0)
    loss_ent.backward()
    assert theta_ent.grad is not None and torch.isfinite(theta_ent.grad)
    assert shift.grad is not None and torch.isfinite(shift.grad)
    # entropic_loss(pnl + c) = entropic_loss(pnl) - c  =>  d(loss)/d(shift) = -1.
    assert abs(float(shift.grad) - (-1.0)) < 1e-4
