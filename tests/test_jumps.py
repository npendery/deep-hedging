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


# ---------------------------------------------------------------------------
# Task 9.3: Merton martingale gate (validates compensator sign)
# ---------------------------------------------------------------------------

def test_simulate_merton_martingale_with_compensator():
    # mu=r=0 -> drift=0; discount factor exp(-r*T)=1. With the compensator,
    # E[S_T] must equal s0 despite the negative-mean jumps.
    cfg = ExperimentConfig(
        s0=100.0, r=0.0, mu=None, sigma=0.2, maturity=1.0, n_steps=50,
        jump_intensity=5.0, jump_mean=-0.1, jump_std=0.15,
    )
    g = torch.Generator().manual_seed(7)
    n_paths = 400_000
    paths = simulate_merton(cfg, n_paths=n_paths, generator=g)
    S_T = paths.S[:, -1]
    mean_ST = S_T.mean().item()
    stderr = (S_T.std(unbiased=True) / (n_paths ** 0.5)).item()
    # discounted E[S_T] ~= s0 within ~3*stderr (r=0 so no discounting needed).
    assert abs(mean_ST - cfg.s0) < 3 * stderr


def test_simulate_merton_martingale_breaks_without_compensator():
    # Sanity that the test above is real: dropping the compensator term biases
    # the mean below s0 by far more than the MC error. (Documents the sign.)
    import math as _math
    cfg = ExperimentConfig(
        s0=100.0, r=0.0, mu=None, sigma=0.2, maturity=1.0, n_steps=50,
        jump_intensity=5.0, jump_mean=-0.1, jump_std=0.15,
    )
    g = torch.Generator().manual_seed(7)
    paths = simulate_merton(cfg, n_paths=400_000, generator=g)
    biased_mean = paths.S[:, -1].mean().item()
    # The compensator over the whole horizon is exp(lambda*(e^{m+s^2/2}-1)*T):
    k_bar = _math.exp(cfg.jump_mean + 0.5 * cfg.jump_std ** 2) - 1.0
    no_comp_factor = _math.exp(cfg.jump_intensity * k_bar * cfg.maturity)
    # If the compensator were missing, the mean would be ~ s0/no_comp_factor.
    # With it present, mean ~= s0 -> well above that biased value.
    assert biased_mean > cfg.s0 * no_comp_factor * 1.01
