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
