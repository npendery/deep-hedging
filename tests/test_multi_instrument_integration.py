import torch

from deephedge.benchmarks import make_bs_delta_vega_strategy
from deephedge.config import ExperimentConfig
from deephedge.instruments import EuropeanOption
from deephedge.portfolio import build_instr_prices, simulate_pnl
from deephedge.pricing.black_scholes import bs_price
from deephedge.simulators.base import get_simulator


def test_simulate_pnl_two_instruments_gbm_runs_and_is_finite():
    cfg = ExperimentConfig(
        s0=100.0, k=100.0, r=0.0, q=0.0, sigma=0.2,
        maturity=30 / 252, n_steps=10, model="gbm", cost=0.001,
        instruments=("underlying", "option"), device="cpu",
    )
    gen = torch.Generator(device="cpu").manual_seed(0)
    paths = get_simulator("gbm")(cfg, 2048, gen)

    # sold call (the liability) and the hedge option (the second instrument)
    sold = EuropeanOption(strike=cfg.k, maturity=cfg.maturity, kind="call")
    hedge = EuropeanOption(strike=cfg.k, maturity=cfg.maturity, kind="call")

    premium = float(bs_price(torch.tensor(cfg.s0), cfg.k, cfg.maturity, cfg.r, cfg.sigma,
                             q=cfg.q, kind="call"))
    instr_prices = build_instr_prices(cfg, paths, hedge)
    assert instr_prices.shape == (2048, cfg.n_steps + 1, 2)

    strat = make_bs_delta_vega_strategy(cfg, hedge)
    result = simulate_pnl(strat, paths, cfg, sold, premium, instr_prices)

    assert result.pnl.shape == (2048,)
    assert torch.isfinite(result.pnl).all()
    # sanity: with a real premium the mean hedged P&L is not pathological
    assert abs(result.pnl.mean().item()) < cfg.s0


def test_simulate_pnl_two_instruments_heston_runs_and_is_finite():
    cfg = ExperimentConfig(
        s0=100.0, k=100.0, r=0.0, q=0.0,
        v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7,
        maturity=30 / 252, n_steps=10, model="heston", cost=0.0,
        instruments=("underlying", "option"), device="cpu",
    )
    gen = torch.Generator(device="cpu").manual_seed(1)
    paths = get_simulator("heston")(cfg, 1024, gen)

    sold = EuropeanOption(strike=cfg.k, maturity=cfg.maturity, kind="call")
    hedge = EuropeanOption(strike=cfg.k, maturity=cfg.maturity, kind="call")

    from deephedge.pricing.heston import heston_price_cm
    premium = float(heston_price_cm(cfg, K=cfg.k, tau=cfg.maturity, kind="call"))
    instr_prices = build_instr_prices(cfg, paths, hedge)
    assert instr_prices.shape == (1024, cfg.n_steps + 1, 2)

    strat = make_bs_delta_vega_strategy(cfg, hedge)
    result = simulate_pnl(strat, paths, cfg, sold, premium, instr_prices)

    assert result.pnl.shape == (1024,)
    assert torch.isfinite(result.pnl).all()
