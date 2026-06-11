"""Regenerate the two money-shot figures for a regime (spec §15).

Usage:
    python -m scripts.make_figures <regime> <outdir>
"""
from __future__ import annotations

import argparse
import dataclasses
import os

from deephedge.evaluate import plot_hedge_ratio, plot_pnl_distribution
from deephedge.train import train
from experiments.regimes import build_regime
from scripts.run_experiment import _pnl_by_strategy


def make_figures(name: str, outdir: str, *, cfg_overrides: dict | None = None) -> dict:
    """Train a hedger for the regime and write only the two charts."""
    os.makedirs(outdir, exist_ok=True)
    cfg = build_regime(name)
    if cfg_overrides:
        cfg = dataclasses.replace(cfg, **cfg_overrides)

    hedger, _ = train(cfg)

    pnl_path = os.path.join(outdir, "pnl_distribution.png")
    band_path = os.path.join(outdir, "hedge_ratio.png")
    plot_pnl_distribution(_pnl_by_strategy(cfg, hedger), pnl_path)
    plot_hedge_ratio(cfg, hedger, band_path)
    return {"pnl_distribution": pnl_path, "hedge_ratio": band_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate deep-hedging figures.")
    parser.add_argument("regime")
    parser.add_argument("outdir")
    args = parser.parse_args()
    figs = make_figures(args.regime, args.outdir)
    for k, v in figs.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
