# tests/simulators/test_heston.py
import math
import torch

from deephedge.config import ExperimentConfig
from deephedge.simulators.base import Paths
from deephedge.simulators.heston import simulate_heston


def _cfg(**kw) -> ExperimentConfig:
    base = dict(
        s0=100.0, k=100.0, r=0.0, q=0.0, mu=None,
        maturity=1.0, n_steps=50,
        v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7,
        model="heston", device="cpu",
    )
    base.update(kw)
    return ExperimentConfig(**base)


def test_shapes_and_initial_conditions():
    cfg = _cfg(n_steps=50)
    gen = torch.Generator(device="cpu").manual_seed(0)
    n_paths = 1000

    paths = simulate_heston(cfg, n_paths, gen)

    assert isinstance(paths, Paths)
    assert paths.S.shape == (n_paths, cfg.n_steps + 1)
    assert paths.V is not None
    assert paths.V.shape == (n_paths, cfg.n_steps + 1)
    # dt and times metadata
    assert math.isclose(paths.dt, cfg.dt, rel_tol=0, abs_tol=1e-12)
    assert paths.times.shape == (cfg.n_steps + 1,)
    assert math.isclose(paths.times[0].item(), 0.0, abs_tol=1e-12)
    assert math.isclose(paths.times[-1].item(), cfg.maturity, abs_tol=1e-9)
    # initial conditions: every path starts at s0 / v0
    assert torch.allclose(paths.S[:, 0], torch.full((n_paths,), cfg.s0))
    assert torch.allclose(paths.V[:, 0], torch.full((n_paths,), cfg.v0))
    # finite everywhere
    assert torch.isfinite(paths.S).all()
    assert torch.isfinite(paths.V).all()


def test_reproducibility_same_seed_same_paths():
    cfg = _cfg(n_steps=20)
    n_paths = 500

    gen_a = torch.Generator(device="cpu").manual_seed(1234)
    gen_b = torch.Generator(device="cpu").manual_seed(1234)
    paths_a = simulate_heston(cfg, n_paths, gen_a)
    paths_b = simulate_heston(cfg, n_paths, gen_b)

    assert torch.equal(paths_a.S, paths_b.S)
    assert torch.equal(paths_a.V, paths_b.V)

    # A different seed must produce different paths (beyond the fixed t=0 column).
    gen_c = torch.Generator(device="cpu").manual_seed(9999)
    paths_c = simulate_heston(cfg, n_paths, gen_c)
    assert not torch.equal(paths_a.S[:, 1:], paths_c.S[:, 1:])
    assert not torch.equal(paths_a.V[:, 1:], paths_c.V[:, 1:])


def test_discounted_expectation_is_martingale():
    # Under risk-neutral drift = r = 0, the discounted price exp(-r*T)*S_T has
    # mean s0 (spec §13.3). With r = 0 this is simply E[S_T] ≈ s0.
    cfg = _cfg(
        s0=100.0, r=0.0, mu=None,           # mu=None -> drift = r = 0
        maturity=1.0, n_steps=100,
        v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7,
    )
    gen = torch.Generator(device="cpu").manual_seed(7)
    n_paths = 200_000

    paths = simulate_heston(cfg, n_paths, gen)
    s_t = paths.S[:, -1]
    discount = math.exp(-cfg.r * cfg.maturity)
    disc_st = discount * s_t

    mean = disc_st.mean().item()
    stderr = (disc_st.std(unbiased=True) / math.sqrt(n_paths)).item()

    # Feller condition here: 2*kappa*theta = 2*1.5*0.04 = 0.12 >= xi^2 = 0.25 is
    # FALSE (violated), so full-truncation discretization carries a small bias.
    # With 200k paths and 100 steps the discounted mean stays within ~3.5 standard
    # errors of s0; this is the spec §13.3 martingale gate.
    assert abs(mean - cfg.s0) < 3.5 * stderr, (
        f"discounted mean {mean:.4f} not within 3.5*stderr ({3.5 * stderr:.4f}) of "
        f"s0={cfg.s0}"
    )
