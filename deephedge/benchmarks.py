"""Analytic / non-learned hedging strategies (spec §10).

Each factory returns a *Strategy*: a callable
``strategy(state: StepState) -> holdings`` with shape ``(B, n_instruments)``.
``make_bs_delta_vega_strategy`` is provided by the multi-instrument section.
"""
from __future__ import annotations

import torch

from deephedge.config import ExperimentConfig
from deephedge.portfolio import StepState


def make_no_hedge_strategy():
    """Hold nothing: collect premium, pay payoff. Context baseline."""

    def strategy(state: StepState) -> torch.Tensor:
        return torch.zeros_like(state.prev_holdings)

    return strategy


from deephedge.pricing.black_scholes import bs_delta  # noqa: E402


def make_bs_delta_strategy(cfg: ExperimentConfig):
    """BS delta hedge of the short call: hold +N(d1) units of the underlying.

    Uses ``cfg.sigma`` (true vol under GBM), ``cfg.r``, ``cfg.q``, the hedge
    option strike (defaults to ``cfg.k``), and ``tau`` taken from ``StepState``.
    Only the underlying leg (column 0) is set; any further legs are zero.
    """
    strike = cfg.hedge_option_strike if cfg.hedge_option_strike is not None else cfg.k

    def strategy(state: StepState) -> torch.Tensor:
        S_i = state.S
        tau = torch.as_tensor(state.tau, dtype=S_i.dtype, device=S_i.device)
        delta = bs_delta(
            S_i,
            torch.as_tensor(strike, dtype=S_i.dtype, device=S_i.device),
            tau,
            cfg.r,
            cfg.sigma,
            q=cfg.q,
            kind="call",
        )
        holdings = torch.zeros_like(state.prev_holdings)
        holdings[:, 0] = delta
        return holdings

    return strategy
