from deephedge.instruments import EuropeanOption


def test_european_option_fields_and_default_kind():
    opt = EuropeanOption(strike=100.0, maturity=0.5)
    assert opt.strike == 100.0
    assert opt.maturity == 0.5
    assert opt.kind == "call"


def test_european_option_put_kind():
    opt = EuropeanOption(strike=90.0, maturity=1.0, kind="put")
    assert opt.kind == "put"


import torch

from deephedge.instruments import payoff


def test_call_payoff_on_small_tensor():
    opt = EuropeanOption(strike=100.0, maturity=0.5, kind="call")
    S_T = torch.tensor([80.0, 100.0, 120.0])
    out = payoff(opt, S_T)
    expected = torch.tensor([0.0, 0.0, 20.0])
    assert torch.allclose(out, expected)
    assert out.shape == S_T.shape


def test_put_payoff_on_small_tensor():
    opt = EuropeanOption(strike=100.0, maturity=0.5, kind="put")
    S_T = torch.tensor([80.0, 100.0, 120.0])
    out = payoff(opt, S_T)
    expected = torch.tensor([20.0, 0.0, 0.0])
    assert torch.allclose(out, expected)


def test_payoff_is_nonnegative_and_differentiable():
    opt = EuropeanOption(strike=100.0, maturity=0.5, kind="call")
    S_T = torch.tensor([90.0, 110.0], requires_grad=True)
    out = payoff(opt, S_T)
    assert (out >= 0).all()
    out.sum().backward()
    # d/dS relu(S-K) = 1 in-the-money, 0 out-of-the-money
    assert torch.allclose(S_T.grad, torch.tensor([0.0, 1.0]))
