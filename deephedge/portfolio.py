"""Differentiable hedged-P&L engine (spec §9) and instrument marking.

A *Strategy* is a callable ``strategy(state: StepState) -> holdings`` returning
a ``(B, n_instruments)`` tensor of target holdings for the current step.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class StepState:
    step: int
    S: torch.Tensor                 # (B,) underlying price at this step
    V: torch.Tensor | None          # (B,) variance at this step, or None
    tau: float                      # time to option maturity (years) at this step
    prev_holdings: torch.Tensor     # (B, n_instruments) holdings carried in
    instr_prices: torch.Tensor      # (B, n_instruments) instrument prices now


@dataclass
class PnLResult:
    pnl: torch.Tensor       # (n_paths,) terminal hedged P&L
    turnover: torch.Tensor  # (n_paths,) sum_i sum_j |delta_i - delta_{i-1}|
    cost: torch.Tensor      # (n_paths,) total proportional transaction cost
    holdings: torch.Tensor  # (n_paths, n_steps, n_instruments) chosen holdings


from deephedge.simulators.base import Paths  # noqa: E402
from deephedge.config import ExperimentConfig  # noqa: E402
from deephedge.instruments import EuropeanOption  # noqa: E402


def build_instr_prices(
    cfg: ExperimentConfig, paths: Paths, hedge_option: EuropeanOption
) -> torch.Tensor:
    """Stack hedging-instrument prices into ``(n_paths, n_steps+1, n_instruments)``.

    Column 0 is always the underlying ``paths.S`` (col tensor shape ``(..., 1)`` in the
    underlying-only case). ``hedge_option`` is the HEDGE instrument's option (its own
    strike/maturity), NOT the sold/liability option. If ``"option"`` is in
    ``cfg.instruments`` the option leg is marked by the multi-instrument section; here it
    raises ``NotImplementedError`` so the contract surface is stable.
    """
    n_paths, n_steps_p1 = paths.S.shape
    cols = [paths.S]  # col 0 = underlying
    for name in cfg.instruments[1:]:
        if name == "option":
            # Filled by the multi-instrument section: mark the HEDGE option leg each
            # step via mark_option(cfg, paths, hedge_option). Deferred here.
            raise NotImplementedError(
                "option hedging instrument is implemented by the "
                "multi-instrument section (use mark_option to fill this branch)"
            )
        raise ValueError(f"unknown hedging instrument: {name!r}")
    out = torch.stack(cols, dim=-1)  # (n_paths, n_steps+1, n_instruments)
    return out.to(cfg.device)
