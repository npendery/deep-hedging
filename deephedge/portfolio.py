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
