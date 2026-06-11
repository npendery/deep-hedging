# tests/test_benchmarks.py
import torch
from deephedge.config import ExperimentConfig
from deephedge.simulators.gbm import simulate_gbm
from deephedge.instruments import EuropeanOption, payoff
from deephedge.portfolio import build_instr_prices, simulate_pnl
from deephedge.benchmarks import make_no_hedge_strategy


def test_no_hedge_pnl_equals_premium_minus_payoff_exact():
    cfg = ExperimentConfig(
        n_steps=12, n_paths=64, cost=0.05, instruments=("underlying",)
    )
    gen = torch.Generator().manual_seed(3)
    paths = simulate_gbm(cfg, n_paths=64, generator=gen)
    option = EuropeanOption(strike=cfg.k, maturity=cfg.maturity, kind="call")
    instr = build_instr_prices(cfg, paths, option)
    premium = 4.0
    strat = make_no_hedge_strategy()
    res = simulate_pnl(strat, paths, cfg, option, premium, instr)

    expected = premium - payoff(option, paths.S[:, -1])
    # Exact: zero holdings -> zero MTM and zero turnover/cost even at cost=0.05.
    assert torch.allclose(res.pnl, expected, atol=1e-6)
    assert torch.allclose(res.cost, torch.zeros(64), atol=1e-7)
    assert torch.allclose(res.turnover, torch.zeros(64), atol=1e-7)
    assert torch.count_nonzero(res.holdings) == 0
