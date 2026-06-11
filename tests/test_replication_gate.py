"""End-to-end replication gate (spec §13(6)).

Frictionless, fine-grid GBM + BS-delta hedge, premium = BS price, must produce a
hedged P&L tightly concentrated at zero. This is a TOLERANCE gate, not bit-exact:
- The MTM sum is a discrete (left-endpoint) approximation of the stochastic
  integral int delta dS; discretization error -> 0 only as n_steps -> infinity.
- On any finite grid the market is incomplete, so perfect replication is
  unattainable; we expect small grid-dependent deviations (spec §13(6)).
Hence we assert mean(|pnl|) is small and std(pnl) < 0.15 * premium, seeded.
"""
import torch

from deephedge.config import ExperimentConfig
from deephedge.simulators.gbm import simulate_gbm
from deephedge.instruments import EuropeanOption
from deephedge.pricing.black_scholes import bs_price
from deephedge.portfolio import build_instr_prices, simulate_pnl
from deephedge.benchmarks import make_bs_delta_strategy


def test_bs_delta_replicates_on_fine_grid_frictionless():
    cfg = ExperimentConfig(
        s0=100.0, k=100.0, r=0.0, q=0.0, mu=None,   # mu=None -> drift = r (RN)
        maturity=30 / 252, n_steps=200,
        sigma=0.2, cost=0.0, instruments=("underlying",),
    )
    gen = torch.Generator().manual_seed(2024)
    n_paths = 20_000
    paths = simulate_gbm(cfg, n_paths=n_paths, generator=gen)

    option = EuropeanOption(strike=cfg.k, maturity=cfg.maturity, kind="call")
    instr = build_instr_prices(cfg, paths, option)

    # Premium = risk-neutral BS price at inception (ATM).
    premium = float(
        bs_price(
            torch.tensor(cfg.s0), torch.tensor(cfg.k),
            torch.tensor(cfg.maturity), cfg.r, cfg.sigma, q=cfg.q, kind="call",
        )
    )
    assert premium > 0

    strat = make_bs_delta_strategy(cfg)
    res = simulate_pnl(strat, paths, cfg, option, premium, instr)

    pnl = res.pnl.detach()
    mean_abs = pnl.abs().mean().item()
    std = pnl.std().item()

    # Mean P&L hugs zero (replication is unbiased in the continuous limit).
    assert abs(pnl.mean().item()) < 0.05 * premium, (
        f"mean P&L {pnl.mean().item():.4f} too large vs premium {premium:.4f}"
    )
    # Tail/variance collapses on the fine grid: std well under 15% of premium.
    assert std < 0.15 * premium, (
        f"std(pnl)={std:.4f} not below 0.15*premium={0.15 * premium:.4f}"
    )
    # No-cost run accrues zero cost; the strategy actually traded.
    assert torch.allclose(res.cost, torch.zeros(n_paths), atol=1e-6)
    assert res.turnover.mean().item() > 0
    # Mean absolute hedging error is a small fraction of premium.
    assert mean_abs < 0.4 * premium
