# tests/test_portfolio.py
import torch
from deephedge.config import ExperimentConfig
from deephedge.simulators.gbm import simulate_gbm
from deephedge.instruments import EuropeanOption
from deephedge.instruments import payoff
from deephedge.portfolio import StepState, PnLResult, build_instr_prices, simulate_pnl


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



def _constant_unit_strategy(state):
    # always hold exactly 1 unit of every instrument
    return torch.ones_like(state.prev_holdings)


def test_simulate_pnl_mtm_telescopes_no_cost():
    # With cost=0 and a constant unit holding in the underlying only, the MTM
    # sum telescopes to (S_N - S_0). So pnl = premium + (S_N - S_0) - payoff.
    cfg = ExperimentConfig(
        n_steps=10, n_paths=16, cost=0.0, instruments=("underlying",)
    )
    gen = torch.Generator().manual_seed(1)
    paths = simulate_gbm(cfg, n_paths=16, generator=gen)
    option = EuropeanOption(strike=cfg.k, maturity=cfg.maturity, kind="call")
    instr = build_instr_prices(cfg, paths, option)
    premium = 3.5
    res = simulate_pnl(
        _constant_unit_strategy, paths, cfg, option, premium, instr
    )
    S0 = paths.S[:, 0]
    S_N = paths.S[:, -1]
    expected = premium + (S_N - S0) - payoff(option, S_N)
    assert res.pnl.shape == (16,)
    assert torch.allclose(res.pnl, expected, atol=1e-5)
    # zero cost with cost=0
    assert torch.allclose(res.cost, torch.zeros(16), atol=1e-7)
    # holdings record: (n_paths, n_steps, n_instruments)
    assert res.holdings.shape == (16, cfg.n_steps, 1)


def test_simulate_pnl_shapes():
    cfg = ExperimentConfig(
        n_steps=7, n_paths=5, cost=0.0, instruments=("underlying",)
    )
    gen = torch.Generator().manual_seed(2)
    paths = simulate_gbm(cfg, n_paths=5, generator=gen)
    option = EuropeanOption(strike=cfg.k, maturity=cfg.maturity, kind="call")
    instr = build_instr_prices(cfg, paths, option)
    res = simulate_pnl(_constant_unit_strategy, paths, cfg, option, 1.0, instr)
    assert res.pnl.shape == (5,)
    assert res.turnover.shape == (5,)
    assert res.cost.shape == (5,)
    assert res.holdings.shape == (5, 7, 1)


def _hold_one_then_two(state):
    # step 0: from prev (0) -> 1 ; steps>=1: hold 2  (one further trade at step 1)
    if state.step == 0:
        return torch.ones_like(state.prev_holdings)
    return 2.0 * torch.ones_like(state.prev_holdings)


def test_simulate_pnl_proportional_cost_exact():
    # Two trades happen: step0 |1-0|=1 priced at S_0; step1 |2-1|=1 priced at S_1.
    # Steps 2..N-1 trade |2-2|=0 -> no further cost.
    cfg = ExperimentConfig(
        n_steps=4, n_paths=3, cost=0.01, instruments=("underlying",)
    )
    gen = torch.Generator().manual_seed(7)
    paths = simulate_gbm(cfg, n_paths=3, generator=gen)
    option = EuropeanOption(strike=cfg.k, maturity=cfg.maturity, kind="call")
    instr = build_instr_prices(cfg, paths, option)
    res = simulate_pnl(_hold_one_then_two, paths, cfg, option, 0.0, instr)

    S0 = paths.S[:, 0]
    S1 = paths.S[:, 1]
    expected_cost = cfg.cost * (S0 * 1.0 + S1 * 1.0)
    assert torch.allclose(res.cost, expected_cost, atol=1e-5)
    # turnover counts |Δδ| summed over steps: 1 (step0) + 1 (step1) = 2 per path
    assert torch.allclose(res.turnover, torch.full((3,), 2.0), atol=1e-6)


def test_simulate_pnl_no_cost_when_rate_zero():
    cfg = ExperimentConfig(
        n_steps=4, n_paths=3, cost=0.0, instruments=("underlying",)
    )
    gen = torch.Generator().manual_seed(7)
    paths = simulate_gbm(cfg, n_paths=3, generator=gen)
    option = EuropeanOption(strike=cfg.k, maturity=cfg.maturity, kind="call")
    instr = build_instr_prices(cfg, paths, option)
    res = simulate_pnl(_hold_one_then_two, paths, cfg, option, 0.0, instr)
    assert torch.allclose(res.cost, torch.zeros(3), atol=1e-7)
    # turnover is independent of the cost rate
    assert torch.allclose(res.turnover, torch.full((3,), 2.0), atol=1e-6)
