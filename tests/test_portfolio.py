# tests/test_portfolio.py
import torch
from deephedge.portfolio import StepState, PnLResult


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
