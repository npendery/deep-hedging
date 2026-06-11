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
