# tests/test_portfolio.py
import torch
from deephedge.config import ExperimentConfig
from deephedge.simulators.gbm import simulate_gbm
from deephedge.instruments import EuropeanOption
from deephedge.portfolio import StepState, PnLResult, build_instr_prices


def test_step_state_holds_fields():
    B, m = 4, 1
    s = torch.full((B,), 100.0)
    holdings = torch.zeros(B, m)
    prices = torch.full((B, m), 100.0)
    state = StepState(
        step=0, S=s, V=None, tau=1.0,
        prev_holdings=holdings, instr_prices=prices,
    )
    assert state.step == 0
    assert state.V is None
    assert state.tau == 1.0
    assert torch.equal(state.S, s)
    assert torch.equal(state.prev_holdings, holdings)
    assert state.instr_prices.shape == (B, m)


def test_pnl_result_holds_fields():
    B = 4
    pnl = torch.zeros(B)
    turnover = torch.zeros(B)
    cost = torch.zeros(B)
    holdings = torch.zeros(B, 5, 1)
    res = PnLResult(pnl=pnl, turnover=turnover, cost=cost, holdings=holdings)
    assert res.pnl.shape == (B,)
    assert res.turnover.shape == (B,)
    assert res.cost.shape == (B,)
    assert res.holdings.shape == (B, 5, 1)


def test_build_instr_prices_underlying_only_is_S():
    cfg = ExperimentConfig(n_steps=5, n_paths=8, instruments=("underlying",))
    gen = torch.Generator().manual_seed(0)
    paths = simulate_gbm(cfg, n_paths=8, generator=gen)
    option = EuropeanOption(strike=cfg.k, maturity=cfg.maturity, kind="call")
    prices = build_instr_prices(cfg, paths, option)
    # shape: (n_paths, n_steps+1, n_instruments)
    assert prices.shape == (8, cfg.n_steps + 1, 1)
    # underlying column equals the simulated path exactly
    assert torch.equal(prices[:, :, 0], paths.S)


def test_build_instr_prices_option_branch_not_implemented():
    cfg = ExperimentConfig(
        n_steps=5, n_paths=8, instruments=("underlying", "option")
    )
    gen = torch.Generator().manual_seed(0)
    paths = simulate_gbm(cfg, n_paths=8, generator=gen)
    option = EuropeanOption(strike=cfg.k, maturity=cfg.maturity, kind="call")
    try:
        build_instr_prices(cfg, paths, option)
        raised = False
    except NotImplementedError:
        raised = True
    assert raised, "option leg must raise NotImplementedError in this section"
