import torch

from deephedge.benchmarks import make_bs_delta_vega_strategy
from deephedge.config import ExperimentConfig
from deephedge.instruments import EuropeanOption
from deephedge.portfolio import StepState


def _state(cfg, S, tau, prev=None, instr_prices=None):
    B = S.shape[0]
    if prev is None:
        prev = torch.zeros(B, cfg.n_instruments)
    if instr_prices is None:
        instr_prices = torch.zeros(B, cfg.n_instruments)
    # step index is informational for analytic strategies; tau drives the Greeks.
    return StepState(
        step=0, S=S, V=None, tau=tau, prev_holdings=prev, instr_prices=instr_prices,
    )


def test_delta_vega_strategy_returns_B_by_2():
    cfg = ExperimentConfig(
        s0=100.0, k=100.0, r=0.0, q=0.0, sigma=0.2,
        maturity=1.0, n_steps=10, model="gbm",
        instruments=("underlying", "option"), device="cpu",
    )
    # hedge option: different strike so its vega differs from the sold call's
    hedge = EuropeanOption(strike=110.0, maturity=cfg.maturity, kind="call")
    strat = make_bs_delta_vega_strategy(cfg, hedge)

    S = torch.tensor([90.0, 100.0, 110.0])
    holdings = strat(_state(cfg, S, tau=0.5))

    assert holdings.shape == (3, 2)
    assert torch.isfinite(holdings).all()


def test_delta_vega_strategy_falls_back_to_delta_when_hedge_vega_vanishes():
    # At tau -> 0 the hedge option's vega -> 0; b must be 0 and a -> sold-call delta.
    from deephedge.pricing.black_scholes import bs_delta

    cfg = ExperimentConfig(
        s0=100.0, k=100.0, r=0.0, q=0.0, sigma=0.2,
        maturity=1.0, n_steps=10, model="gbm",
        instruments=("underlying", "option"), device="cpu",
    )
    hedge = EuropeanOption(strike=100.0, maturity=cfg.maturity, kind="call")
    strat = make_bs_delta_vega_strategy(cfg, hedge)

    S = torch.tensor([95.0, 100.0, 105.0])
    holdings = strat(_state(cfg, S, tau=1e-8))

    # option holding collapses to ~0 (no usable vega), underlying ~ sold-call delta
    assert torch.allclose(holdings[:, 1], torch.zeros(3), atol=1e-6)
    delta_sold = bs_delta(S, cfg.k, 1e-8, cfg.r, cfg.sigma, q=cfg.q)
    assert torch.allclose(holdings[:, 0], delta_sold, atol=1e-4)


def test_delta_vega_strategy_neutralizes_delta_and_vega():
    from deephedge.pricing.black_scholes import bs_delta, bs_vega

    cfg = ExperimentConfig(
        s0=100.0, k=100.0, r=0.01, q=0.0, sigma=0.2,
        maturity=1.0, n_steps=10, model="gbm",
        instruments=("underlying", "option"), device="cpu",
    )
    # longer-dated, different-strike hedge option => non-degenerate vega at this step
    hedge = EuropeanOption(strike=105.0, maturity=1.5, kind="call")
    strat = make_bs_delta_vega_strategy(cfg, hedge)

    S = torch.tensor([90.0, 100.0, 110.0, 120.0])
    tau_s = 0.5                                   # sold-call remaining time
    holdings = strat(_state(cfg, S, tau=tau_s))
    a = holdings[:, 0]
    b = holdings[:, 1]

    t_now = cfg.maturity - tau_s
    tau_h = max(hedge.maturity - t_now, 0.0)      # = 1.5 - 0.5 = 1.0

    # sold-call Greeks (we are SHORT one call -> liability has these Greeks)
    delta_s = bs_delta(S, cfg.k, tau_s, cfg.r, cfg.sigma, q=cfg.q)
    vega_s = bs_vega(S, cfg.k, tau_s, cfg.r, cfg.sigma, q=cfg.q)
    # hedge-option Greeks
    delta_h = bs_delta(S, hedge.strike, tau_h, cfg.r, cfg.sigma, q=cfg.q)
    vega_h = bs_vega(S, hedge.strike, tau_h, cfg.r, cfg.sigma, q=cfg.q)

    # net delta of the hedged book = a*1 + b*delta_h - delta_s  (underlying delta = 1)
    net_delta = a * 1.0 + b * delta_h - delta_s
    # net vega = a*0 + b*vega_h - vega_s   (underlying vega = 0)
    net_vega = b * vega_h - vega_s

    assert torch.allclose(net_delta, torch.zeros_like(net_delta), atol=1e-5)
    assert torch.allclose(net_vega, torch.zeros_like(net_vega), atol=1e-5)
    # the option leg actually carries weight here (not the degenerate b=0 case)
    assert b.abs().mean() > 1e-3
