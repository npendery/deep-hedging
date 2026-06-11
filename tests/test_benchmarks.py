# tests/test_benchmarks.py
import torch
from deephedge.config import ExperimentConfig
from deephedge.simulators.gbm import simulate_gbm
from deephedge.instruments import EuropeanOption, payoff
from deephedge.portfolio import build_instr_prices, simulate_pnl
from deephedge.portfolio import StepState, build_instr_prices, simulate_pnl
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


from deephedge.pricing.black_scholes import bs_delta
from deephedge.benchmarks import make_bs_delta_strategy


def test_bs_delta_strategy_matches_bs_delta():
    cfg = ExperimentConfig(
        n_steps=10, n_paths=32, sigma=0.2, r=0.0, q=0.0,
        instruments=("underlying",),
    )
    strat = make_bs_delta_strategy(cfg)
    S_i = torch.linspace(80.0, 120.0, 32)
    prev = torch.zeros(32, 1)
    tau = 0.05
    state = StepState(
        step=2, S=S_i, V=None, tau=tau,
        prev_holdings=prev, instr_prices=S_i.unsqueeze(-1),
    )
    holdings = strat(state)
    assert holdings.shape == (32, 1)
    expected = bs_delta(
        S_i, torch.tensor(cfg.k), torch.tensor(tau),
        cfg.r, cfg.sigma, q=cfg.q, kind="call",
    )
    assert torch.allclose(holdings[:, 0], expected, atol=1e-5)


def test_bs_delta_strategy_deep_itm_near_one():
    cfg = ExperimentConfig(sigma=0.2, r=0.0, q=0.0, instruments=("underlying",))
    strat = make_bs_delta_strategy(cfg)
    S_i = torch.full((4,), 200.0)  # deep ITM call -> delta ~ 1
    state = StepState(
        step=0, S=S_i, V=None, tau=0.05,
        prev_holdings=torch.zeros(4, 1), instr_prices=S_i.unsqueeze(-1),
    )
    holdings = strat(state)
    assert torch.all(holdings[:, 0] > 0.99)
