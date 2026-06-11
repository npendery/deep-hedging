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


from deephedge.pricing.black_scholes import bs_vega  # noqa: E402


def make_bs_delta_vega_strategy(cfg: ExperimentConfig, option):
    """Strategy neutralizing the sold call's BS delta AND vega with (underlying, option).

    Holdings (a, b) in (underlying, hedge-option) solve the 2x2 system:
        b * vega_h        = vega_s          (vega:  underlying has zero vega)
        a + b * delta_h   = delta_s         (delta: underlying has unit delta)
    => b = vega_s / vega_h ; a = delta_s - b * delta_h.
    Greeks use sigma=cfg.sigma under the risk-neutral measure. tau for each leg is the
    step's remaining time (state.tau) for the sold call and
    (option.maturity - (cfg.maturity - state.tau)) clamped >= 0 for the hedge option
    (so a longer-dated hedge option keeps positive time-to-maturity at the sold call's
    expiry). When vega_h is ~0 the system is singular in b; fall back to b=0, a=delta_s.
    """

    def strategy(state: StepState) -> torch.Tensor:
        S = state.S                                  # (B,)
        tau_s = float(state.tau)                     # sold-call time-to-maturity
        t_now = cfg.maturity - tau_s                 # elapsed calendar time
        tau_h = max(option.maturity - t_now, 0.0)    # hedge-option time-to-maturity

        delta_s = bs_delta(S, cfg.k, tau_s, cfg.r, cfg.sigma, q=cfg.q)
        vega_s = bs_vega(S, cfg.k, tau_s, cfg.r, cfg.sigma, q=cfg.q)
        delta_h = bs_delta(S, option.strike, tau_h, cfg.r, cfg.sigma, q=cfg.q)
        vega_h = bs_vega(S, option.strike, tau_h, cfg.r, cfg.sigma, q=cfg.q)

        # Use a spot-normalized vega threshold: max vega is S*sqrt(tau)/sqrt(2pi).
        # At tau->0 this collapses to ~S*sqrt(tau)*0.4, so normalising by S makes the
        # threshold scale-invariant. eps=1e-3 catches tau<=1e-6 and deep ITM/OTM options
        # while leaving mid-maturity, near-ATM options usable.
        eps = 1e-3
        vega_h_norm = vega_h / S.clamp_min(1e-8)    # normalise by spot
        usable = vega_h_norm.abs() > eps
        b = torch.where(usable, vega_s / vega_h.clamp_min(eps * S.clamp_min(1e-8)), torch.zeros_like(vega_h))
        a = delta_s - b * delta_h
        return torch.stack([a, b], dim=-1)           # (B, 2)

    return strategy


from deephedge.hedger import build_features  # noqa: E402


def make_nn_strategy(hedger, cfg: ExperimentConfig):
    """Learned policy: build_features(...) -> Hedger.forward -> holdings.

    Keeps the autograd graph intact so gradients flow from terminal P&L back
    through the trajectory to ``hedger`` parameters (spec §9, §13(7)).
    """

    def strategy(state: StepState) -> torch.Tensor:
        # Normalize tau to (T - t_i)/T and pass cfg.k as the moneyness strike, per the
        # build_features contract. state.S may be (B,) — build_features reshapes to columns.
        features = build_features(
            state.S,
            state.tau / cfg.maturity,
            state.prev_holdings,
            state.V,
            k_norm=cfg.k,
        )
        return hedger(features)

    return strategy
