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


def test_put_call_parity():
    # C - P = S*exp(-q*tau) - K*exp(-r*tau). With nonzero r, q, off-ATM strikes.
    gen = torch.Generator().manual_seed(0)
    S = 80.0 + 40.0 * torch.rand(16, generator=gen)   # spreads around 100
    K = torch.tensor(100.0)
    r, q, sigma, tau = 0.03, 0.01, 0.25, 0.5
    call = bs_price(S, K, tau, r, sigma, q=q, kind="call")
    put = bs_price(S, K, tau, r, sigma, q=q, kind="put")
    lhs = call - put
    rhs = S * math.exp(-q * tau) - K * math.exp(-r * tau)
    assert torch.allclose(lhs, rhs, atol=1e-5)
