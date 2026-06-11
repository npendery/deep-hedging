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
