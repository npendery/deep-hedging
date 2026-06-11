# tests/simulators/test_get_simulator_heston.py
import torch

from deephedge.config import ExperimentConfig
from deephedge.simulators.base import Paths, get_simulator
from deephedge.simulators.heston import simulate_heston


def test_get_simulator_returns_heston_callable():
    sim = get_simulator("heston")
    assert sim is simulate_heston

    cfg = ExperimentConfig(
        s0=100.0, maturity=1.0, n_steps=10,
        v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7,
        model="heston", device="cpu",
    )
    gen = torch.Generator(device="cpu").manual_seed(0)
    paths = sim(cfg, 100, gen)
    assert isinstance(paths, Paths)
    assert paths.S.shape == (100, cfg.n_steps + 1)
    assert paths.V is not None and paths.V.shape == (100, cfg.n_steps + 1)
