"""Tests for experiments/regimes.py (Group 12, Task 12.8)."""
import pytest

from deephedge.config import ExperimentConfig
from experiments.regimes import REGIMES, build_regime


def test_regimes_registry_names():
    assert set(REGIMES.keys()) == {
        "gbm", "gbm_costs", "heston_costs", "bates_costs", "multi_instrument"
    }


def test_build_regime_returns_experiment_config():
    for name in REGIMES:
        cfg = build_regime(name)
        assert isinstance(cfg, ExperimentConfig)


def test_regime_model_and_cost_settings():
    assert build_regime("gbm").model == "gbm"
    assert build_regime("gbm").cost == 0.0

    assert build_regime("gbm_costs").model == "gbm"
    assert build_regime("gbm_costs").cost > 0.0

    assert build_regime("heston_costs").model == "heston"
    assert build_regime("heston_costs").cost > 0.0

    assert build_regime("bates_costs").model == "bates"
    assert build_regime("bates_costs").cost > 0.0
    assert build_regime("bates_costs").jump_intensity > 0.0  # Bates has jumps

    mi = build_regime("multi_instrument")
    assert mi.instruments == ("underlying", "option")
    assert mi.n_instruments == 2
    assert mi.model == "heston"


def test_build_regime_unknown_raises_value_error():
    with pytest.raises(ValueError, match="nope"):
        build_regime("nope")
