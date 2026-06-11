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


# ---------------------------------------------------------------------------
# Task 9.4: Merton MC vs closed-form Poisson-weighted BS call price
# ---------------------------------------------------------------------------

def test_simulate_merton_matches_closed_form_call():
    import math as _math
    from math import erf, lgamma

    def _norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + erf(x / _math.sqrt(2.0)))

    def _bs_call(s0, K, T, r, sigma):
        if sigma <= 0.0 or T <= 0.0:
            return max(s0 - K * _math.exp(-r * T), 0.0)
        vol = sigma * _math.sqrt(T)
        d1 = (_math.log(s0 / K) + (r + 0.5 * sigma ** 2) * T) / vol
        d2 = d1 - vol
        return s0 * _norm_cdf(d1) - K * _math.exp(-r * T) * _norm_cdf(d2)

    def _merton_call(s0, K, T, r, sigma, lam, m, s, n_terms=40):
        k_bar = _math.exp(m + 0.5 * s ** 2) - 1.0
        lam_p = lam * (1.0 + k_bar)             # lam' = lam*exp(m+0.5 s^2)
        price = 0.0
        for n in range(n_terms):
            sigma_n = _math.sqrt(sigma ** 2 + n * s ** 2 / T)
            r_n = r - lam * k_bar + n * (m + 0.5 * s ** 2) / T
            log_w = -lam_p * T + n * _math.log(lam_p * T) - lgamma(n + 1)
            weight = _math.exp(log_w)
            price += weight * _bs_call(s0, K, T, r_n, sigma_n)
        return price

    cfg = ExperimentConfig(
        s0=100.0, k=100.0, r=0.03, mu=None, sigma=0.2, maturity=1.0, n_steps=100,
        jump_intensity=1.0, jump_mean=-0.1, jump_std=0.15,
    )
    closed = _merton_call(
        cfg.s0, cfg.k, cfg.maturity, cfg.r, cfg.sigma,
        cfg.jump_intensity, cfg.jump_mean, cfg.jump_std, n_terms=40,
    )

    g = torch.Generator().manual_seed(11)
    n_paths = 500_000
    paths = simulate_merton(cfg, n_paths=n_paths, generator=g)
    S_T = paths.S[:, -1]
    disc = _math.exp(-cfg.r * cfg.maturity)
    payoff = torch.clamp(S_T - cfg.k, min=0.0) * disc
    mc_price = payoff.mean().item()
    mc_stderr = (payoff.std(unbiased=True) / (n_paths ** 0.5)).item()

    # closed-form must lie inside the MC 99% CI (z=2.58).
    assert abs(mc_price - closed) < 2.58 * mc_stderr, (
        f"MC={mc_price:.4f} closed={closed:.4f} stderr={mc_stderr:.4f}"
    )


# ---------------------------------------------------------------------------
# Task 9.5: simulate_merton reduces to simulate_gbm when jump_intensity=0
# ---------------------------------------------------------------------------

from deephedge.simulators.base import get_simulator
from deephedge.simulators.gbm import simulate_gbm
from deephedge.simulators.jumps import simulate_bates


def test_simulate_merton_zero_intensity_matches_gbm_in_distribution():
    cfg = ExperimentConfig(
        s0=100.0, r=0.01, mu=None, sigma=0.2, maturity=1.0, n_steps=50,
        jump_intensity=0.0, jump_mean=-0.1, jump_std=0.15,
    )
    n_paths = 300_000
    g_m = torch.Generator().manual_seed(123)
    g_g = torch.Generator().manual_seed(123)
    merton = simulate_merton(cfg, n_paths=n_paths, generator=g_m)
    gbm = simulate_gbm(cfg, n_paths=n_paths, generator=g_g)

    st_m = merton.S[:, -1]
    # cast GBM (float32) to float64 for a like-dtype comparison
    st_g = gbm.S[:, -1].to(torch.float64)
    # Means and stds of S_T agree to <1% relative (both are the same GBM law).
    assert abs(st_m.mean().item() - st_g.mean().item()) / st_g.mean().item() < 0.01
    assert abs(st_m.std().item() - st_g.std().item()) / st_g.std().item() < 0.02
    # Quantile match at the 5% and 95% tails (<1.5% relative).
    qs = torch.tensor([0.05, 0.95], dtype=st_m.dtype)
    qm = torch.quantile(st_m, qs)
    qg = torch.quantile(st_g, qs)
    assert torch.all((qm - qg).abs() / qg < 0.015)


# ---------------------------------------------------------------------------
# Task 9.6: simulate_bates shapes and variance state
# ---------------------------------------------------------------------------

def test_simulate_bates_shapes_and_variance_state():
    cfg = ExperimentConfig(
        s0=100.0, r=0.0, mu=None, maturity=1.0, n_steps=50,
        v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7,
        jump_intensity=2.0, jump_mean=-0.1, jump_std=0.15,
    )
    g = torch.Generator().manual_seed(0)
    n_paths = 5_000
    paths = simulate_bates(cfg, n_paths=n_paths, generator=g)
    assert isinstance(paths, Paths)
    assert paths.S.shape == (n_paths, cfg.n_steps + 1)
    assert paths.V is not None and paths.V.shape == (n_paths, cfg.n_steps + 1)
    assert torch.allclose(paths.S[:, 0], torch.full((n_paths,), cfg.s0, dtype=paths.S.dtype))
    assert torch.allclose(paths.V[:, 0], torch.full((n_paths,), cfg.v0, dtype=paths.V.dtype))
    assert torch.all(paths.S > 0.0)             # jumps act on log-price -> S > 0
    assert abs(paths.dt - cfg.dt) < 1e-12
    assert paths.times.shape == (cfg.n_steps + 1,)


# ---------------------------------------------------------------------------
# Task 9.7: simulate_bates martingale check
# ---------------------------------------------------------------------------

def test_simulate_bates_martingale_with_compensator():
    cfg = ExperimentConfig(
        s0=100.0, r=0.0, mu=None, maturity=1.0, n_steps=100,
        v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7,
        jump_intensity=5.0, jump_mean=-0.1, jump_std=0.15,
    )
    g = torch.Generator().manual_seed(21)
    n_paths = 500_000
    paths = simulate_bates(cfg, n_paths=n_paths, generator=g)
    S_T = paths.S[:, -1]
    mean_ST = S_T.mean().item()
    stderr = (S_T.std(unbiased=True) / (n_paths ** 0.5)).item()
    # r=0 -> discount factor 1; mean must sit within ~3*stderr of s0.
    assert abs(mean_ST - cfg.s0) < 3 * stderr


# ---------------------------------------------------------------------------
# Task 9.8: get_simulator resolves "merton" and "bates"
# ---------------------------------------------------------------------------

def test_get_simulator_registers_merton_and_bates():
    g = torch.Generator().manual_seed(0)

    cfg_m = ExperimentConfig(
        model="merton", s0=100.0, sigma=0.2, maturity=1.0, n_steps=20,
        jump_intensity=2.0, jump_mean=-0.1, jump_std=0.15,
    )
    sim_m = get_simulator("merton")
    paths_m = sim_m(cfg_m, 1_000, g)
    assert paths_m.S.shape == (1_000, cfg_m.n_steps + 1)
    assert paths_m.V is None
    # Same callable identity as the direct function.
    assert sim_m is simulate_merton

    cfg_b = ExperimentConfig(
        model="bates", s0=100.0, maturity=1.0, n_steps=20,
        v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7,
        jump_intensity=2.0, jump_mean=-0.1, jump_std=0.15,
    )
    sim_b = get_simulator("bates")
    paths_b = sim_b(cfg_b, 1_000, g)
    assert paths_b.S.shape == (1_000, cfg_b.n_steps + 1)
    assert paths_b.V is not None and paths_b.V.shape == (1_000, cfg_b.n_steps + 1)
    assert sim_b is simulate_bates


# ---------------------------------------------------------------------------
# Task 9.9: Device-agnostic sanity
# ---------------------------------------------------------------------------

def test_jump_simulators_respect_cfg_device():
    cfg = ExperimentConfig(
        model="merton", device="cpu", s0=100.0, sigma=0.2, maturity=0.5, n_steps=10,
        jump_intensity=1.0, jump_mean=-0.05, jump_std=0.1,
    )
    g = torch.Generator().manual_seed(3)
    pm = simulate_merton(cfg, 256, g)
    assert pm.S.device.type == "cpu"
    assert pm.times.device.type == "cpu"

    cfg_b = ExperimentConfig(
        model="bates", device="cpu", s0=100.0, maturity=0.5, n_steps=10,
        v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7,
        jump_intensity=1.0, jump_mean=-0.05, jump_std=0.1,
    )
    pb = simulate_bates(cfg_b, 256, g)
    assert pb.S.device.type == "cpu"
    assert pb.V.device.type == "cpu"
