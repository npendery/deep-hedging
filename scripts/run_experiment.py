"""Train + evaluate + save the two figures for one named regime (spec §14/§15).

Usage:
    python -m scripts.run_experiment <regime> <outdir>
"""
from __future__ import annotations

import argparse
import dataclasses
import os

import torch

from deephedge.benchmarks import (
    make_bs_delta_strategy,
    make_bs_delta_vega_strategy,
    make_nn_strategy,
    make_no_hedge_strategy,
)
from deephedge.evaluate import (
    _hedge_option,
    _model_premium,
    empirical_cvar,
    evaluate,
    metrics_table,
    plot_hedge_ratio,
    plot_pnl_distribution,
)
from deephedge.instruments import EuropeanOption
from deephedge.portfolio import build_instr_prices, simulate_pnl
from deephedge.simulators.base import get_simulator
from deephedge.train import train
from experiments.regimes import build_regime

_EVAL_SEED_OFFSET = 10_000


def _pnl_by_strategy(cfg, hedger) -> dict:
    """Per-path P&L + CVaR_95 per strategy on the SAME fresh paths evaluate() uses.

    Threads the two distinct options exactly as evaluate(): the `liability` (sold) option
    drives the simulate_pnl payoff/premium; the `hedge` option is marked into instr_prices
    col1 and feeds the delta+vega benchmark (same instance both places).
    """
    device = torch.device(cfg.device)
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.seed + _EVAL_SEED_OFFSET)
    paths = get_simulator(cfg.model)(cfg, cfg.n_paths, gen)
    liability = EuropeanOption(cfg.k, cfg.maturity)
    hedge = _hedge_option(cfg) if "option" in cfg.instruments else None
    premium = _model_premium(cfg, liability)
    instr_prices = build_instr_prices(cfg, paths, hedge)

    strategies = {
        "nn": make_nn_strategy(hedger, cfg),
        "bs_delta": make_bs_delta_strategy(cfg),
        "no_hedge": make_no_hedge_strategy(),
    }
    if cfg.n_instruments == 2:
        strategies["bs_delta_vega"] = make_bs_delta_vega_strategy(cfg, hedge)

    enriched: dict = {}
    with torch.no_grad():
        for name, strat in strategies.items():
            pnl = simulate_pnl(strat, paths, cfg, liability, premium, instr_prices).pnl
            enriched[name] = {"pnl": pnl, "cvar_95": empirical_cvar(pnl, 0.95)}
    return enriched


def run_experiment(name: str, outdir: str, *, cfg_overrides: dict | None = None) -> dict:
    """Train, evaluate, and write the metrics table + two figures for a regime."""
    os.makedirs(outdir, exist_ok=True)
    cfg = build_regime(name)
    if cfg_overrides:
        cfg = dataclasses.replace(cfg, **cfg_overrides)

    hedger, _history = train(cfg)
    results = evaluate(cfg, hedger)
    table = metrics_table(results)

    with open(os.path.join(outdir, "metrics.md"), "w") as fh:
        fh.write(f"# Regime: {name}\n\n{table}\n")

    pnl_path = os.path.join(outdir, "pnl_distribution.png")
    band_path = os.path.join(outdir, "hedge_ratio.png")
    plot_pnl_distribution(_pnl_by_strategy(cfg, hedger), pnl_path)
    plot_hedge_ratio(cfg, hedger, band_path)

    return {
        "results": results,
        "table": table,
        "figures": {"pnl_distribution": pnl_path, "hedge_ratio": band_path},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train+evaluate a deep-hedging regime.")
    parser.add_argument("regime")
    parser.add_argument("outdir")
    args = parser.parse_args()
    out = run_experiment(args.regime, args.outdir)
    print(out["table"])


if __name__ == "__main__":
    main()
