"""Per-regime ExperimentConfig builders (spec §14).

Regimes: gbm, gbm_costs, heston_costs, bates_costs, multi_instrument. Each builder
returns a fully-populated ExperimentConfig; `build_regime(name)` dispatches by name.
"""
from __future__ import annotations

from typing import Callable

from deephedge.config import ExperimentConfig

_COST = 0.005  # proportional transaction-cost rate for the friction regimes


def _gbm() -> ExperimentConfig:
    return ExperimentConfig(
        model="gbm", sigma=0.2, cost=0.0,
        maturity=30 / 252, n_steps=30,
        loss="cvar", alpha=0.95,
        n_paths=100_000, batch_size=8192,
        epochs=50, steps_per_epoch=50, lr=1e-3, seed=0,
    )


def _gbm_costs() -> ExperimentConfig:
    return ExperimentConfig(
        model="gbm", sigma=0.2, cost=_COST,
        maturity=30 / 252, n_steps=30,
        loss="cvar", alpha=0.95,
        n_paths=100_000, batch_size=8192,
        epochs=50, steps_per_epoch=50, lr=1e-3, seed=0,
    )


def _heston_costs() -> ExperimentConfig:
    return ExperimentConfig(
        model="heston", cost=_COST,
        v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7,
        maturity=30 / 252, n_steps=30,
        loss="cvar", alpha=0.95,
        n_paths=100_000, batch_size=8192,
        epochs=50, steps_per_epoch=50, lr=1e-3, seed=0,
    )


def _bates_costs() -> ExperimentConfig:
    return ExperimentConfig(
        model="bates", cost=_COST,
        v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7,
        jump_intensity=1.0, jump_mean=-0.1, jump_std=0.15,
        maturity=30 / 252, n_steps=30,
        loss="cvar", alpha=0.95,
        n_paths=100_000, batch_size=8192,
        epochs=50, steps_per_epoch=50, lr=1e-3, seed=0,
    )


def _multi_instrument() -> ExperimentConfig:
    return ExperimentConfig(
        model="heston", cost=_COST,
        v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7,
        instruments=("underlying", "option"),
        hedge_option_strike=100.0, hedge_option_maturity=60 / 252,
        maturity=30 / 252, n_steps=30,
        loss="cvar", alpha=0.95,
        n_paths=100_000, batch_size=8192,
        epochs=50, steps_per_epoch=50, lr=1e-3, seed=0,
    )


REGIMES: dict[str, Callable[[], ExperimentConfig]] = {
    "gbm": _gbm,
    "gbm_costs": _gbm_costs,
    "heston_costs": _heston_costs,
    "bates_costs": _bates_costs,
    "multi_instrument": _multi_instrument,
}


def build_regime(name: str) -> ExperimentConfig:
    """Build the ExperimentConfig for a named regime (spec §14)."""
    try:
        return REGIMES[name]()
    except KeyError:
        known = ", ".join(sorted(REGIMES))
        raise ValueError(f"Unknown regime {name!r}; known regimes: {known}")
