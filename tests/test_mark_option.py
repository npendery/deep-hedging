import torch

from deephedge.config import ExperimentConfig
from deephedge.instruments import EuropeanOption, mark_option, payoff
from deephedge.simulators.base import Paths


def _gbm_paths(cfg: ExperimentConfig, n_paths: int, seed: int = 0) -> Paths:
    """Tiny deterministic GBM-style path bundle for marking tests."""
    gen = torch.Generator(device=cfg.device).manual_seed(seed)
    n = cfg.n_steps
    times = torch.linspace(0.0, cfg.maturity, n + 1, device=cfg.device)
    z = torch.randn(n_paths, n, generator=gen, device=cfg.device)
    incr = (cfg.drift - 0.5 * cfg.sigma**2) * cfg.dt + cfg.sigma * (cfg.dt**0.5) * z
    log_s = torch.empty(n_paths, n + 1, device=cfg.device)
    log_s[:, 0] = torch.log(torch.tensor(cfg.s0, device=cfg.device))
    log_s[:, 1:] = log_s[:, :1] + torch.cumsum(incr, dim=1)
    return Paths(S=log_s.exp(), V=None, dt=cfg.dt, times=times)


def test_mark_option_shape_and_terminal_equals_payoff_gbm():
    cfg = ExperimentConfig(
        s0=100.0, k=100.0, r=0.0, q=0.0, sigma=0.2,
        maturity=1.0, n_steps=8, model="gbm", device="cpu",
    )
    option = EuropeanOption(strike=100.0, maturity=cfg.maturity, kind="call")
    paths = _gbm_paths(cfg, n_paths=64, seed=1)

    marks = mark_option(cfg, paths, option)

    # shape is (n_paths, n_steps+1)
    assert marks.shape == (64, cfg.n_steps + 1)
    assert torch.isfinite(marks).all()
    # an ATM call mark is strictly positive before expiry, and < spot
    assert (marks[:, 0] > 0).all()
    assert (marks[:, 0] < paths.S[:, 0]).all()
    # terminal column == intrinsic payoff within BS tau->0 pricing tolerance
    terminal = marks[:, -1]
    intrinsic = payoff(option, paths.S[:, -1])
    assert torch.allclose(terminal, intrinsic, atol=1e-4)


def test_mark_option_t0_column_matches_bs_price_with_cfg_sigma():
    # At t0 every path is at s0; the mark must equal bs_price(s0, K, T, r, cfg.sigma, q).
    from deephedge.pricing.black_scholes import bs_price

    cfg = ExperimentConfig(
        s0=100.0, k=100.0, r=0.01, q=0.0, sigma=0.25,
        maturity=0.5, n_steps=5, model="gbm", device="cpu",
    )
    option = EuropeanOption(strike=110.0, maturity=cfg.maturity, kind="call")
    paths = _gbm_paths(cfg, n_paths=16, seed=2)

    marks = mark_option(cfg, paths, option)

    expected0 = bs_price(
        paths.S[:, 0], option.strike, option.maturity, cfg.r, cfg.sigma, q=cfg.q,
        kind=option.kind,
    )
    assert torch.allclose(marks[:, 0], expected0, atol=1e-6)


def test_mark_option_is_differentiable_in_spot():
    # MTM gains across the option leg must backprop into the spot path (spec §9).
    cfg = ExperimentConfig(
        s0=100.0, k=100.0, r=0.0, q=0.0, sigma=0.2,
        maturity=1.0, n_steps=4, model="gbm", device="cpu",
    )
    option = EuropeanOption(strike=100.0, maturity=cfg.maturity, kind="call")

    times = torch.linspace(0.0, cfg.maturity, cfg.n_steps + 1)
    S = torch.full((3, cfg.n_steps + 1), 100.0, requires_grad=True)
    paths = Paths(S=S, V=None, dt=cfg.dt, times=times)

    marks = mark_option(cfg, paths, option)
    marks.sum().backward()

    assert S.grad is not None
    assert torch.isfinite(S.grad).all()
    # in-the-money / ATM call mark increases with spot -> positive sensitivity pre-expiry
    assert (S.grad[:, 0] > 0).all()


def test_mark_option_tau_clamped_when_option_matures_before_horizon():
    # A hedge option maturing at T/2 must have tau=0 (intrinsic) for all later nodes.
    cfg = ExperimentConfig(
        s0=100.0, k=100.0, r=0.0, q=0.0, sigma=0.2,
        maturity=1.0, n_steps=10, model="gbm", device="cpu",
    )
    option = EuropeanOption(strike=100.0, maturity=0.5, kind="call")
    paths = _gbm_paths(cfg, n_paths=32, seed=3)

    marks = mark_option(cfg, paths, option)

    # nodes with t_i >= 0.5 have tau=0 -> mark equals intrinsic payoff exactly-ish
    late = paths.times >= 0.5
    intrinsic_late = payoff(option, paths.S[:, late])
    assert torch.allclose(marks[:, late], intrinsic_late, atol=1e-4)
    # an early node (t=0, tau=0.5) is worth strictly more than intrinsic (time value)
    assert (marks[:, 0] > payoff(option, paths.S[:, 0]) - 1e-6).all()
    assert marks[:, 0].mean() > payoff(option, paths.S[:, 0]).mean()


import numpy as np

from deephedge.instruments import _heston_implied_vol
from deephedge.pricing.heston import heston_price_cm
from deephedge.pricing.black_scholes import bs_price as _bs_price


def _heston_cfg(**kw) -> ExperimentConfig:
    base = dict(
        s0=100.0, k=100.0, r=0.0, q=0.0,
        v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7,
        maturity=1.0, n_steps=8, model="heston", device="cpu",
    )
    base.update(kw)
    return ExperimentConfig(**base)


def test_heston_implied_vol_reprices_heston_t0_price():
    # The inverted BS implied vol, plugged back into bs_price at t0, must reproduce the
    # Heston model price of the hedge option to tight tolerance.
    cfg = _heston_cfg()
    option = EuropeanOption(strike=100.0, maturity=cfg.maturity, kind="call")

    sigma_impl = _heston_implied_vol(cfg, option)
    assert sigma_impl > 0.0
    assert np.isfinite(sigma_impl)

    heston_px = heston_price_cm(cfg, K=option.strike, tau=option.maturity, kind="call")
    bs_px = _bs_price(
        torch.tensor(cfg.s0), option.strike, option.maturity, cfg.r, sigma_impl, q=cfg.q,
        kind="call",
    ).item()
    assert abs(bs_px - heston_px) < 5e-3


def test_mark_option_heston_terminal_equals_payoff_and_t0_matches_heston():
    cfg = _heston_cfg()
    option = EuropeanOption(strike=100.0, maturity=cfg.maturity, kind="call")
    # build a simple Heston-style path bundle (variance held flat for the test paths;
    # mark_option uses the FROZEN proxy vol, so V is not consumed by the mark)
    gen = torch.Generator(device="cpu").manual_seed(5)
    n = cfg.n_steps
    times = torch.linspace(0.0, cfg.maturity, n + 1)
    z = torch.randn(40, n, generator=gen)
    incr = (cfg.drift - 0.5 * cfg.v0) * cfg.dt + (cfg.v0**0.5) * (cfg.dt**0.5) * z
    log_s = torch.empty(40, n + 1)
    log_s[:, 0] = torch.log(torch.tensor(cfg.s0))
    log_s[:, 1:] = log_s[:, :1] + torch.cumsum(incr, dim=1)
    V = torch.full((40, n + 1), cfg.v0)
    paths = Paths(S=log_s.exp(), V=V, dt=cfg.dt, times=times)

    marks = mark_option(cfg, paths, option)

    assert marks.shape == (40, n + 1)
    # terminal column == intrinsic payoff (BS at tau=0)
    assert torch.allclose(marks[:, -1], payoff(option, paths.S[:, -1]), atol=1e-4)
    # t0 mark (all paths at s0) reproduces the Heston model price within proxy tolerance
    heston_px = heston_price_cm(cfg, K=option.strike, tau=option.maturity, kind="call")
    assert abs(marks[:, 0].mean().item() - heston_px) < 5e-3
