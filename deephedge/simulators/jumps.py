"""Merton (GBM + jumps) and Bates (Heston + jumps) simulators.

Compound-Poisson log-jumps with the Merton/Bates risk-neutral compensator so the
discounted spot is a martingale under mu = r (spec §5.3).

## Poisson randomness routing

`torch.poisson(rate_tensor, generator=generator)` is used for Poisson counts.
As of PyTorch 2.x, `torch.poisson` does accept the `generator` keyword argument
and routes the sampling through that generator, so all randomness (Poisson counts
AND Gaussian jump sizes) flows through the caller-supplied generator. The implementation
below passes `generator=generator` to every random call to guarantee reproducibility.
"""
import math

import torch

from deephedge.config import ExperimentConfig
from deephedge.simulators.base import Paths


def _sample_jumps(
    cfg: ExperimentConfig, n_paths: int, generator: torch.Generator
) -> torch.Tensor:
    """Aggregate per-step log-jump increment for each path: (n_paths,).

    n ~ Poisson(jump_intensity*dt) jumps per step, each Y_k ~ N(jump_mean, jump_std^2);
    the sum of n iid normals is N(n*jump_mean, n*jump_std^2), so we draw the count then a
    single normal, avoiding materializing individual jumps. When n=0 the increment is
    exactly 0 (both the mean term and sqrt(n) scale term vanish).

    Poisson randomness: torch.poisson accepts a generator kwarg in PyTorch 2.x, so the
    Poisson draw is routed through the caller-supplied generator for reproducibility.
    """
    device = torch.device(cfg.device)
    rate = torch.full(
        (n_paths,), cfg.jump_intensity * cfg.dt, device=device
    )
    counts = torch.poisson(rate, generator=generator)  # float tensor of non-neg ints
    z = torch.randn(n_paths, generator=generator, device=device)
    jumps = counts * cfg.jump_mean + torch.sqrt(counts) * cfg.jump_std * z
    return jumps


def _jump_compensator(cfg: ExperimentConfig) -> float:
    """Per-step risk-neutral drift compensator:
    jump_intensity*(exp(jump_mean + 0.5*jump_std^2) - 1)*dt  (Merton 1976 / Bates 1996).

    This is subtracted from the log-price drift each step so that the discounted spot
    price is a martingale under the risk-neutral measure (drift = r).
    """
    k_bar = math.exp(cfg.jump_mean + 0.5 * cfg.jump_std ** 2) - 1.0
    return cfg.jump_intensity * k_bar * cfg.dt


def simulate_merton(
    cfg: ExperimentConfig, n_paths: int, generator: torch.Generator
) -> Paths:
    """GBM log-Euler + compound-Poisson jumps + risk-neutral compensator (spec §5.3).

    log S_{i+1} = log S_i + (drift - 0.5*sigma^2)*dt - comp + sigma*sqrt(dt)*Z + J_i
    where comp = jump_intensity*(exp(jump_mean + 0.5*jump_std^2) - 1)*dt.

    The compensator ensures E[S_T] = S_0 * exp(drift * T) (martingale under drift=r).
    V is None — Merton has no variance process.
    """
    device = torch.device(cfg.device)
    dt = cfg.dt
    n = cfg.n_steps
    comp = _jump_compensator(cfg)
    diffusion_drift = (cfg.drift - 0.5 * cfg.sigma ** 2) * dt - comp

    log_s = torch.empty(n_paths, n + 1, device=device)
    log_s[:, 0] = math.log(cfg.s0)
    sqrt_dt = math.sqrt(dt)
    for i in range(n):
        z = torch.randn(n_paths, generator=generator, device=device)
        jumps = _sample_jumps(cfg, n_paths=n_paths, generator=generator)
        log_s[:, i + 1] = (
            log_s[:, i] + diffusion_drift + cfg.sigma * sqrt_dt * z + jumps
        )

    S = torch.exp(log_s)
    times = torch.linspace(0.0, cfg.maturity, n + 1, device=device)
    return Paths(S=S, V=None, dt=dt, times=times)


def simulate_bates(
    cfg: ExperimentConfig, n_paths: int, generator: torch.Generator
) -> Paths:
    """Heston full-truncation Euler + compound-Poisson jumps + compensator (spec §5.3).

    V_plus       = max(V_i, 0)
    V_{i+1}      = V_i + kappa*(theta - V_plus)*dt + xi*sqrt(V_plus)*sqrt(dt)*Z2
    log S_{i+1}  = log S_i + (drift - 0.5*V_plus)*dt - comp
                            + sqrt(V_plus)*sqrt(dt)*Z1 + J_i
    Z1 = Za, Z2 = rho*Za + sqrt(1-rho^2)*Zb  (Cholesky); carried V may go negative.

    The jump compensator (same as Merton) ensures the discounted spot is a martingale.
    V is returned (not None) — Bates has a stochastic variance process.
    """
    device = torch.device(cfg.device)
    dt = cfg.dt
    n = cfg.n_steps
    sqrt_dt = math.sqrt(dt)
    comp = _jump_compensator(cfg)
    rho = cfg.rho
    sqrt_1m_rho2 = math.sqrt(max(1.0 - rho ** 2, 0.0))

    log_s = torch.empty(n_paths, n + 1, device=device)
    V = torch.empty(n_paths, n + 1, device=device)
    log_s[:, 0] = math.log(cfg.s0)
    V[:, 0] = cfg.v0

    for i in range(n):
        za = torch.randn(n_paths, generator=generator, device=device)
        zb = torch.randn(n_paths, generator=generator, device=device)
        z1 = za
        z2 = rho * za + sqrt_1m_rho2 * zb
        jumps = _sample_jumps(cfg, n_paths=n_paths, generator=generator)

        v_prev = V[:, i]
        v_plus = torch.clamp(v_prev, min=0.0)
        sqrt_v_plus = torch.sqrt(v_plus)

        # carried variance state is NOT truncated (full-truncation scheme).
        V[:, i + 1] = (
            v_prev + cfg.kappa * (cfg.theta - v_plus) * dt
            + cfg.xi * sqrt_v_plus * sqrt_dt * z2
        )
        log_s[:, i + 1] = (
            log_s[:, i] + (cfg.drift - 0.5 * v_plus) * dt - comp
            + sqrt_v_plus * sqrt_dt * z1 + jumps
        )

    S = torch.exp(log_s)
    times = torch.linspace(0.0, cfg.maturity, n + 1, device=device)
    return Paths(S=S, V=V, dt=dt, times=times)
