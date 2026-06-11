import dataclasses

import pytest

from deephedge.config import ExperimentConfig


def test_defaults():
    cfg = ExperimentConfig()
    assert cfg.s0 == 100.0
    assert cfg.k == 100.0
    assert cfg.r == 0.0
    assert cfg.q == 0.0
    assert cfg.mu is None
    assert cfg.maturity == pytest.approx(30 / 252)
    assert cfg.n_steps == 30
    assert cfg.sigma == 0.2
    assert cfg.model == "gbm"
    assert cfg.loss == "cvar"
    assert cfg.alpha == 0.95
    assert cfg.instruments == ("underlying",)
    assert cfg.device == "cpu"
    assert cfg.seed == 0


def test_frozen():
    cfg = ExperimentConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.s0 = 50.0  # type: ignore[misc]
