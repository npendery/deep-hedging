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


def test_dt_is_maturity_over_n_steps():
    cfg = ExperimentConfig(maturity=1.0, n_steps=4)
    assert cfg.dt == pytest.approx(0.25)


def test_drift_defaults_to_r_when_mu_none():
    cfg = ExperimentConfig(r=0.03, mu=None)
    assert cfg.drift == pytest.approx(0.03)


def test_drift_uses_mu_when_set():
    cfg = ExperimentConfig(r=0.03, mu=0.10)
    assert cfg.drift == pytest.approx(0.10)


def test_n_instruments_counts_instruments():
    assert ExperimentConfig().n_instruments == 1
    assert ExperimentConfig(instruments=("underlying", "option")).n_instruments == 2
