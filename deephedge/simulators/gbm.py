from __future__ import annotations

import torch

from deephedge.config import ExperimentConfig
from deephedge.simulators.base import Paths


def simulate_gbm(
    cfg: ExperimentConfig,
    n_paths: int,
    generator: torch.Generator,
) -> Paths:
    """Exact log-Euler GBM:  S_{i+1} = S_i * exp((drift - 0.5*sigma^2)dt + sigma*sqrt(dt)*Z).

    Returns Paths with S of shape (n_paths, n_steps+1), V=None, the float step dt,
    and the time grid linspace(0, maturity, n_steps+1). All tensors on cfg.device.
    """
    device = torch.device(cfg.device)
    n_steps = cfg.n_steps
    dt = cfg.dt
    drift = cfg.drift
    sigma = cfg.sigma

    times = torch.linspace(0.0, cfg.maturity, n_steps + 1, device=device)

    # standard normal increments Z: one per (path, step)
    Z = torch.randn(n_paths, n_steps, generator=generator, device=device)

    log_increments = (drift - 0.5 * sigma * sigma) * dt + sigma * (dt ** 0.5) * Z

    log_S = torch.empty(n_paths, n_steps + 1, device=device)
    log_S[:, 0] = torch.log(torch.tensor(cfg.s0, device=device))
    log_S[:, 1:] = log_S[:, :1] + torch.cumsum(log_increments, dim=1)

    S = torch.exp(log_S)
    return Paths(S=S, V=None, dt=dt, times=times)
