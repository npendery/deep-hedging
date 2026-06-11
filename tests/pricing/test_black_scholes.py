import math

import torch

from deephedge.pricing.black_scholes import _d1_d2, bs_price, bs_delta, bs_gamma, bs_vega


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


def test_call_delta_atm_half_boundary():
    # ATM r=q=0 sigma=0.2 tau=1 -> delta = N(0.1) ~ 0.5398
    S = torch.tensor(100.0)
    K = torch.tensor(100.0)
    delta = bs_delta(S, K, 1.0, 0.0, 0.2, kind="call")
    assert torch.allclose(delta, torch.tensor(0.5398), atol=1e-3)


def test_call_delta_deep_itm_and_otm():
    K = torch.tensor(100.0)
    r, sigma, tau = 0.0, 0.2, 1.0
    itm = bs_delta(torch.tensor(1000.0), K, tau, r, sigma, kind="call")
    otm = bs_delta(torch.tensor(1.0), K, tau, r, sigma, kind="call")
    assert itm.item() > 0.999            # deep ITM call delta -> 1
    assert otm.item() < 1e-3             # deep OTM call delta -> 0


def test_put_delta_equals_call_delta_minus_one():
    # With q=0: put delta = call delta - 1 (= -N(-d1)).
    S = torch.tensor(100.0)
    K = torch.tensor(100.0)
    r, sigma, tau = 0.0, 0.2, 1.0
    cd = bs_delta(S, K, tau, r, sigma, kind="call")
    pd = bs_delta(S, K, tau, r, sigma, kind="put")
    assert torch.allclose(pd, cd - 1.0, atol=1e-6)


def test_vega_positive_and_known_value():
    # ATM r=q=0 sigma=0.2 tau=1 -> vega = S*phi(d1)*sqrt(tau) = 39.6953
    S = torch.tensor(100.0)
    K = torch.tensor(100.0)
    vega = bs_vega(S, K, 1.0, 0.0, 0.2)
    assert vega.item() > 0.0
    assert torch.allclose(vega, torch.tensor(39.6953), atol=1e-3)


def test_gamma_positive_and_known_value():
    # ATM r=q=0 sigma=0.2 tau=1 -> gamma = phi(d1)/(S*sigma*sqrt(tau)) = 0.0198476
    S = torch.tensor(100.0)
    K = torch.tensor(100.0)
    gamma = bs_gamma(S, K, 1.0, 0.0, 0.2)
    assert gamma.item() > 0.0
    assert torch.allclose(gamma, torch.tensor(0.0198476), atol=1e-6)


def test_vega_gamma_relation():
    # vega = gamma * S^2 * sigma * tau (standard identity, q=0)
    S = torch.tensor(120.0)
    K = torch.tensor(100.0)
    r, sigma, tau = 0.01, 0.3, 0.7
    vega = bs_vega(S, K, tau, r, sigma)
    gamma = bs_gamma(S, K, tau, r, sigma)
    assert torch.allclose(vega, gamma * S * S * sigma * tau, atol=1e-4)
