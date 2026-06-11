import torch

from deephedge.config import ExperimentConfig
from deephedge.instruments import EuropeanOption, mark_option
from deephedge.portfolio import build_instr_prices
from deephedge.simulators.base import Paths


def _gbm_paths(cfg: ExperimentConfig, n_paths: int, seed: int = 0) -> Paths:
    gen = torch.Generator(device=cfg.device).manual_seed(seed)
    n = cfg.n_steps
    times = torch.linspace(0.0, cfg.maturity, n + 1, device=cfg.device)
    z = torch.randn(n_paths, n, generator=gen, device=cfg.device)
    incr = (cfg.drift - 0.5 * cfg.sigma**2) * cfg.dt + cfg.sigma * (cfg.dt**0.5) * z
    log_s = torch.empty(n_paths, n + 1, device=cfg.device)
    log_s[:, 0] = torch.log(torch.tensor(cfg.s0, device=cfg.device))
    log_s[:, 1:] = log_s[:, :1] + torch.cumsum(incr, dim=1)
    return Paths(S=log_s.exp(), V=None, dt=cfg.dt, times=times)


def test_build_instr_prices_underlying_only_is_single_column():
    cfg = ExperimentConfig(
        s0=100.0, k=100.0, sigma=0.2, maturity=1.0, n_steps=6,
        model="gbm", instruments=("underlying",), device="cpu",
    )
    option = EuropeanOption(strike=cfg.k, maturity=cfg.maturity, kind="call")
    paths = _gbm_paths(cfg, n_paths=32, seed=1)

    prices = build_instr_prices(cfg, paths, option)

    assert prices.shape == (32, cfg.n_steps + 1, 1)
    # column 0 is the underlying spot itself
    assert torch.allclose(prices[:, :, 0], paths.S)


def test_build_instr_prices_two_instruments_has_underlying_and_option():
    cfg = ExperimentConfig(
        s0=100.0, k=100.0, sigma=0.2, maturity=1.0, n_steps=6,
        model="gbm", instruments=("underlying", "option"), device="cpu",
    )
    # hedge option defaults to the sold-call strike/maturity per the contract
    option = EuropeanOption(strike=cfg.k, maturity=cfg.maturity, kind="call")
    paths = _gbm_paths(cfg, n_paths=32, seed=2)

    prices = build_instr_prices(cfg, paths, option)

    # (n_paths, n_steps+1, 2) and NO NotImplementedError
    assert prices.shape == (32, cfg.n_steps + 1, 2)
    # col0 == spot, col1 == mark_option
    assert torch.allclose(prices[:, :, 0], paths.S)
    assert torch.allclose(prices[:, :, 1], mark_option(cfg, paths, option))
    assert torch.isfinite(prices).all()
