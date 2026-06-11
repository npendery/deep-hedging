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
from deephedge.instruments import EuropeanOption, mark_option  # noqa: E402


def build_instr_prices(
    cfg: ExperimentConfig, paths: Paths, hedge_option: EuropeanOption
) -> torch.Tensor:
    """Stack hedging-instrument prices into ``(n_paths, n_steps+1, n_instruments)``.

    ``hedge_option`` is the HEDGE instrument's option (its own strike/maturity), NOT the
    sold/liability option. Column j corresponds to cfg.instruments[j]:
      - "underlying": the spot path paths.S
      - "option":     the marked hedge-option price mark_option(cfg, paths, hedge_option)
    """
    cols = []
    for name in cfg.instruments:
        if name == "underlying":
            cols.append(paths.S)
        elif name == "option":
            cols.append(mark_option(cfg, paths, hedge_option))
        else:
            raise ValueError(
                f"unknown hedging instrument {name!r}; "
                "expected 'underlying' or 'option'"
            )
    out = torch.stack(cols, dim=-1)  # (n_paths, n_steps+1, n_instruments)
    return out.to(cfg.device)


from deephedge.instruments import payoff as _payoff  # noqa: E402


def simulate_pnl(
    strategy,
    paths: Paths,
    cfg: ExperimentConfig,
    option: EuropeanOption,
    premium: float,
    instr_prices: torch.Tensor,
) -> PnLResult:
    """Roll the hedged trajectory forward and accumulate terminal P&L (spec §9).

    ``strategy(state: StepState) -> holdings`` returns ``(B, n_instruments)``.
    Holdings are chosen at steps ``0..N-1`` (no trade is needed at the terminal
    step ``N``). P&L = premium + MTM gains - proportional costs - short-call
    payoff. Fully differentiable in torch w.r.t. anything ``strategy`` depends on.
    """
    n_paths, n_steps_p1, n_instr = instr_prices.shape
    n_steps = n_steps_p1 - 1

    S = paths.S
    V = paths.V
    dt = cfg.dt

    prev = torch.zeros(n_paths, n_instr, device=instr_prices.device)
    pnl = torch.zeros(n_paths, device=instr_prices.device)
    turnover = torch.zeros(n_paths, device=instr_prices.device)
    cost = torch.zeros(n_paths, device=instr_prices.device)
    holdings_log = []

    for i in range(n_steps):
        p_i = instr_prices[:, i, :]       # (n_paths, n_instr)
        p_next = instr_prices[:, i + 1, :]
        tau = cfg.maturity - i * dt       # time to option maturity at step i
        V_i = None if V is None else V[:, i]
        state = StepState(
            step=i,
            S=S[:, i],
            V=V_i,
            tau=tau,
            prev_holdings=prev,
            instr_prices=p_i,
        )
        holdings = strategy(state)        # (n_paths, n_instr)
        holdings_log.append(holdings)

        # MTM gain over [t_i, t_{i+1}] across all instruments
        pnl = pnl + (holdings * (p_next - p_i)).sum(dim=-1)

        # proportional transaction cost at rebalance time t_i
        trade = (holdings - prev).abs()             # (n_paths, n_instr)
        turnover = turnover + trade.sum(dim=-1)
        step_cost = cfg.cost * (p_i * trade).sum(dim=-1)
        cost = cost + step_cost
        pnl = pnl - step_cost

        prev = holdings

    # premium received (short the call) and terminal liability
    pnl = pnl + premium - _payoff(option, S[:, -1])

    holdings_out = torch.stack(holdings_log, dim=1)  # (n_paths, n_steps, n_instr)
    return PnLResult(pnl=pnl, turnover=turnover, cost=cost, holdings=holdings_out)
