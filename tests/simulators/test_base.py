import torch

from deephedge.simulators.base import Paths


def test_paths_holds_fields():
    n_paths, n_steps = 5, 3
    S = torch.zeros(n_paths, n_steps + 1)
    times = torch.linspace(0.0, 1.0, n_steps + 1)
    paths = Paths(S=S, V=None, dt=0.25, times=times)
    assert paths.S.shape == (n_paths, n_steps + 1)
    assert paths.V is None
    assert paths.dt == 0.25
    assert paths.times.shape == (n_steps + 1,)


def test_paths_can_carry_variance():
    n_paths, n_steps = 5, 3
    S = torch.zeros(n_paths, n_steps + 1)
    V = torch.ones(n_paths, n_steps + 1)
    times = torch.linspace(0.0, 1.0, n_steps + 1)
    paths = Paths(S=S, V=V, dt=0.25, times=times)
    assert paths.V is not None
    assert paths.V.shape == (n_paths, n_steps + 1)
