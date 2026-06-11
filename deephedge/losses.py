"""Convex risk-measure losses for deep hedging (spec §6).

We work consistently in loss space ``L = -pnl`` (larger = worse). ``pnl`` is the
profit-positive terminal P&L tensor of shape ``(N,)`` (one entry per Monte-Carlo path).
"""

import math

import torch


def cvar_loss(pnl: torch.Tensor, alpha: float, w: torch.Tensor) -> torch.Tensor:
    """Rockafellar-Uryasev CVaR loss (spec §6.1).

    ``L = -pnl`` ;  ``F(params, w) = w + (1/(1-alpha)) * mean(relu(L - w))``.

    ``w`` is a learnable scalar (``nn.Parameter`` of shape ``()``) passed in by the
    caller. We MINIMIZE ``F`` jointly over ``(network params, w)`` with Adam.

    Justification for the joint optimization is the UPPER-BOUND / recovery property,
    NOT joint convexity (spec §6.1): because ``CVaR(params) = min_w F(params, w)``, for
    any fixed ``w`` we have ``F(params, w) >= CVaR(params)``, hence
    ``min_{params, w} F = min_{params} CVaR``. With a neural-net hedge the problem is
    NON-CONVEX (RU joint convexity needs the loss convex in the decision variable — RU
    2002 Cor. 11 / RU 2000 Thm 2), so do NOT claim global optimality. The
    ``(1 - alpha)`` tail-probability denominator is correct; using ``alpha`` is the
    classic ~19x mis-scaling bug. The ``alpha=0.95`` tail uses only ~5% of paths, so
    gradients are high variance — train with large/representative batches.
    """
    loss = -pnl
    return w + (1.0 / (1.0 - alpha)) * torch.relu(loss - w).mean()


def entropic_loss(pnl: torch.Tensor, lam: float) -> torch.Tensor:
    """Entropic (exponential) risk measure (spec §6.2).

    ``(1/lam) * (logsumexp(-lam*pnl) - log(N))`` with ``N = pnl.shape[0]``. This equals
    ``(1/lam) * log(mean(exp(-lam*pnl)))``: the ``- log(N)`` turns the sum into a mean so
    loss values are comparable across batch sizes (argmin is unaffected, but reporting is
    wrong without it). ``pnl`` is profit-positive, so the ``exp(-lam*pnl)`` exponent
    penalizes losses exponentially. Convex, monotone, cash-invariant (not coherent);
    equivalent to maximizing CARA utility ``u(x) = -exp(-lam*x)`` with ``lam`` = absolute
    risk aversion. ``torch.logsumexp`` is max-shift stabilized (exact). As ``lam -> 0+``
    the loss -> ``-E[pnl]``; the empirical estimator is biased ``O(1/N)`` (Jensen),
    vanishing as ``N -> inf``.
    """
    n = pnl.shape[0]
    log_n = math.log(n)
    return (1.0 / lam) * (torch.logsumexp(-lam * pnl, dim=0) - log_n)
