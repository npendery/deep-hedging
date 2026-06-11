"""Tests for scripts/run_experiment.py and scripts/make_figures.py (Group 12, Tasks 12.9-12.10)."""
import torch

from scripts.run_experiment import run_experiment


def test_run_experiment_trains_evaluates_and_saves(tmp_path):
    # Override the heavy regime defaults down to a few fast steps.
    overrides = dict(
        n_paths=1024, batch_size=1024,
        n_steps=8, epochs=1, steps_per_epoch=3, lr=5e-3, seed=1,
    )
    out = run_experiment("gbm_costs", str(tmp_path), cfg_overrides=overrides)

    # nested results for the three single-instrument strategies
    assert set(out["results"].keys()) == {"nn", "bs_delta", "no_hedge"}
    # markdown table written to disk and returned
    assert (tmp_path / "metrics.md").exists()
    assert out["table"].startswith("|")
    # both figures saved with regime-prefixed names
    pnl_fig = tmp_path / "gbm_costs_pnl_distribution.png"
    band_fig = tmp_path / "gbm_costs_hedge_ratio.png"
    assert pnl_fig.exists() and pnl_fig.stat().st_size > 0
    assert band_fig.exists() and band_fig.stat().st_size > 0
    assert out["figures"]["pnl_distribution"] == str(pnl_fig)
    assert out["figures"]["hedge_ratio"] == str(band_fig)


from scripts.make_figures import make_figures  # noqa: E402


def test_make_figures_writes_both_charts(tmp_path):
    overrides = dict(
        n_paths=1024, batch_size=1024,
        n_steps=8, epochs=1, steps_per_epoch=2, lr=5e-3, seed=2,
    )
    figs = make_figures("gbm_costs", str(tmp_path), cfg_overrides=overrides)

    pnl_fig = tmp_path / "gbm_costs_pnl_distribution.png"
    band_fig = tmp_path / "gbm_costs_hedge_ratio.png"
    assert pnl_fig.exists() and pnl_fig.stat().st_size > 0
    assert band_fig.exists() and band_fig.stat().st_size > 0
    assert figs == {"pnl_distribution": str(pnl_fig), "hedge_ratio": str(band_fig)}
