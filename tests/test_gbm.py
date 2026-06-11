# tests/test_gbm.py
import torch

from deephedge.config import ExperimentConfig
from deephedge.simulators.base import Paths, get_simulator
from deephedge.simulators.gbm import simulate_gbm


def test_gbm_shape_and_initial_condition():
    cfg = ExperimentConfig(s0=100.0, sigma=0.2, maturity=1.0, n_steps=50)
    gen = torch.Generator(device=cfg.device).manual_seed(0)
    n_paths = 1000

    paths = simulate_gbm(cfg, n_paths, gen)

    assert isinstance(paths, Paths)
    assert paths.S.shape == (n_paths, cfg.n_steps + 1)
    assert paths.V is None
    # exact dt and a (n_steps+1,) time grid from 0..maturity
    assert paths.dt == cfg.dt
    assert paths.times.shape == (cfg.n_steps + 1,)
    assert float(paths.times[0]) == 0.0
    assert torch.isclose(paths.times[-1], torch.tensor(cfg.maturity))
    # every path starts at s0
    assert torch.allclose(paths.S[:, 0], torch.full((n_paths,), cfg.s0))
    # all prices strictly positive (exp keeps GBM positive)
    assert torch.all(paths.S > 0)


def test_get_simulator_gbm_dispatches_to_simulate_gbm():
    cfg = ExperimentConfig(s0=100.0, sigma=0.2, maturity=1.0, n_steps=10)
    gen = torch.Generator(device=cfg.device).manual_seed(0)
    sim = get_simulator("gbm")
    # get_simulator's lazy "gbm" branch returns simulate_gbm itself.
    assert sim is simulate_gbm
    paths = sim(cfg, 16, gen)
    assert isinstance(paths, Paths)
    assert paths.S.shape == (16, cfg.n_steps + 1)
    assert paths.V is None
