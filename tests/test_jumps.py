"""Tests for Merton and Bates jump-diffusion simulators (Group 9)."""
import math

import torch

from deephedge.config import ExperimentConfig
from deephedge.simulators.base import Paths
from deephedge.simulators.jumps import _sample_jumps, simulate_merton


def test_sample_jumps_zero_intensity_is_exactly_zero():
    # With jump_intensity=0 every Poisson count is 0, so the aggregate
    # log-jump must be identically zero (no jump contribution).
    cfg = ExperimentConfig(jump_intensity=0.0, jump_mean=-0.1, jump_std=0.15)
    g = torch.Generator().manual_seed(0)
    J = _sample_jumps(cfg, n_paths=10_000, generator=g)
    assert J.shape == (10_000,)
    assert torch.all(J == 0.0)


def test_sample_jumps_mean_matches_compound_poisson():
    # E[J] = E[N]*jump_mean = (lambda*dt)*jump_mean.
    cfg = ExperimentConfig(
        jump_intensity=5.0, jump_mean=-0.1, jump_std=0.15,
        maturity=1.0, n_steps=1,
    )
    g = torch.Generator().manual_seed(1)
    n_paths = 400_000
    J = _sample_jumps(cfg, n_paths=n_paths, generator=g)
    expected_mean = cfg.jump_intensity * cfg.dt * cfg.jump_mean
    # Var[J] = E[N]*(jump_mean^2 + jump_std^2); stderr of the sample mean:
    lam_dt = cfg.jump_intensity * cfg.dt
    var_J = lam_dt * (cfg.jump_mean ** 2 + cfg.jump_std ** 2)
    stderr = math.sqrt(var_J / n_paths)
    assert abs(J.mean().item() - expected_mean) < 4 * stderr


def test_sample_jumps_variance_matches_compound_poisson():
    # Var[J] = E[N]*(jump_mean^2 + jump_std^2)  (compound-Poisson variance).
    cfg = ExperimentConfig(
        jump_intensity=8.0, jump_mean=0.0, jump_std=0.2,
        maturity=1.0, n_steps=1,
    )
    g = torch.Generator().manual_seed(2)
    n_paths = 400_000
    J = _sample_jumps(cfg, n_paths=n_paths, generator=g)
    lam_dt = cfg.jump_intensity * cfg.dt
    expected_var = lam_dt * (cfg.jump_mean ** 2 + cfg.jump_std ** 2)
    sample_var = J.var(unbiased=True).item()
    # 5% relative tolerance at 4e5 paths.
    assert abs(sample_var - expected_var) / expected_var < 0.05


# ---------------------------------------------------------------------------
# Task 9.2: simulate_merton shapes and state
# ---------------------------------------------------------------------------

def test_simulate_merton_shapes_and_state():
    cfg = ExperimentConfig(
        s0=100.0, sigma=0.2, maturity=30 / 252, n_steps=30,
        jump_intensity=2.0, jump_mean=-0.1, jump_std=0.15,
    )
    g = torch.Generator().manual_seed(0)
    n_paths = 5_000
    paths = simulate_merton(cfg, n_paths=n_paths, generator=g)
    assert isinstance(paths, Paths)
    assert paths.S.shape == (n_paths, cfg.n_steps + 1)
    assert paths.V is None                      # Merton has no variance process
    assert paths.times.shape == (cfg.n_steps + 1,)
    assert abs(paths.dt - cfg.dt) < 1e-12
    # First column is exactly s0.
    assert torch.allclose(paths.S[:, 0], torch.full((n_paths,), cfg.s0, dtype=paths.S.dtype))
    # times grid is 0 .. maturity inclusive, evenly spaced.
    assert paths.times[0].item() == 0.0
    assert abs(paths.times[-1].item() - cfg.maturity) < 1e-12
    # Strictly positive prices (jumps act on the log-price -> S stays > 0).
    assert torch.all(paths.S > 0.0)
