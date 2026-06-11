import math

import torch

from deephedge.pricing.black_scholes import _d1_d2, bs_price


def test_d1_d2_atm_unit_vol_one_year():
    # ATM, r=q=0, sigma=0.2, tau=1 -> d1 = +sigma/2*sqrt(tau) = 0.1, d2 = -0.1
    S = torch.tensor(100.0)
    K = torch.tensor(100.0)
    d1, d2, sqrt_tau = _d1_d2(S, K, 1.0, 0.0, 0.2, 0.0)
    assert torch.allclose(d1, torch.tensor(0.1), atol=1e-6)
    assert torch.allclose(d2, torch.tensor(-0.1), atol=1e-6)
    assert torch.allclose(sqrt_tau, torch.tensor(1.0), atol=1e-6)


def test_atm_call_price_textbook_value():
    # Hull textbook value: S=K=100, r=0, sigma=0.2, T=1 -> 7.9656 (computed 7.965567...)
    S = torch.tensor(100.0)
    K = torch.tensor(100.0)
    price = bs_price(S, K, 1.0, 0.0, 0.2, kind="call")
    assert torch.allclose(price, torch.tensor(7.9656), atol=1e-3)
