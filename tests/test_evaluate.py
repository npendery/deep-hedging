"""Tests for deephedge/evaluate.py (Group 12, Tasks 12.1-12.7)."""
import math

import pytest
import torch

from deephedge.config import ExperimentConfig
from deephedge.instruments import EuropeanOption
from deephedge.evaluate import _model_premium


def _tiny_gbm_cfg(**overrides) -> ExperimentConfig:
    base = dict(
        s0=100.0, k=100.0, r=0.0, q=0.0, mu=None,
        maturity=30 / 252, n_steps=8,
        sigma=0.2, model="gbm", loss="cvar", alpha=0.95,
        cost=0.0, instruments=("underlying",),
        n_paths=1024, batch_size=1024,
        epochs=1, steps_per_epoch=1,
        lr=1e-3, seed=0, device="cpu",
    )
    base.update(overrides)
    return ExperimentConfig(**base)


# --- Task 12.1: _model_premium ---

def test_model_premium_gbm_matches_bs_price():
    from deephedge.pricing.black_scholes import bs_price

    cfg = _tiny_gbm_cfg()
    option = EuropeanOption(cfg.k, cfg.maturity)
    prem = _model_premium(cfg, option)

    S0 = torch.tensor(cfg.s0, dtype=torch.float64)
    expected = float(bs_price(S0, cfg.k, cfg.maturity, cfg.r, cfg.sigma, q=cfg.q, kind="call"))

    assert isinstance(prem, float)
    assert math.isclose(prem, expected, rel_tol=1e-9, abs_tol=1e-9)
    # ATM call under r=0, q=0 is positive and strictly below spot.
    assert 0.0 < prem < cfg.s0


def test_model_premium_heston_uses_carr_madan():
    from deephedge.pricing.heston import heston_price_cm

    cfg = _tiny_gbm_cfg(model="heston")
    option = EuropeanOption(cfg.k, cfg.maturity)
    prem = _model_premium(cfg, option)
    expected = float(heston_price_cm(cfg, cfg.k, cfg.maturity, kind="call"))

    assert math.isclose(prem, expected, rel_tol=1e-9, abs_tol=1e-9)
    assert 0.0 < prem < cfg.s0


# --- Task 12.2: empirical_cvar ---

def test_empirical_cvar_matches_cvar_loss_at_optimal_w():
    from deephedge.losses import cvar_loss
    from deephedge.evaluate import empirical_cvar

    gen = torch.Generator().manual_seed(2024)
    # profit-positive P&L sample (so L = -pnl is the loss); large N for a tight match.
    pnl = torch.randn(200_000, generator=gen)
    alpha = 0.95

    cvar = empirical_cvar(pnl, alpha)

    # cvar_loss at the empirical optimum w* = VaR_alpha = quantile(L, alpha).
    losses = -pnl
    w_star = torch.quantile(losses, alpha)
    ru = float(cvar_loss(pnl, alpha, torch.nn.Parameter(w_star.clone())).detach())

    assert isinstance(cvar, float)
    assert abs(cvar - ru) < 1e-5


def test_empirical_cvar_matches_gaussian_closed_form():
    from deephedge.evaluate import empirical_cvar

    # For L ~ N(0,1), CVaR_alpha(L) = phi(z_alpha) / (1 - alpha) with z_alpha = Phi^{-1}(alpha).
    gen = torch.Generator().manual_seed(7)
    pnl = torch.randn(1_000_000, generator=gen)  # L = -pnl is also N(0,1)
    alpha = 0.95

    cvar = empirical_cvar(pnl, alpha)

    z = math.sqrt(2.0) * torch.erfinv(torch.tensor(2.0 * alpha - 1.0)).item()
    phi_z = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    closed_form = phi_z / (1.0 - alpha)  # ~2.0627 for alpha=0.95

    assert abs(cvar - closed_form) < 2e-2


# --- Task 12.3: _strategy_metrics ---

def test_strategy_metrics_keys_and_values():
    from deephedge.portfolio import PnLResult
    from deephedge.evaluate import _strategy_metrics, empirical_cvar

    gen = torch.Generator().manual_seed(5)
    pnl = torch.randn(50_000, generator=gen)
    turnover = torch.full((50_000,), 3.0)
    cost = torch.full((50_000,), 0.25)
    res = PnLResult(pnl=pnl, turnover=turnover, cost=cost, holdings=torch.zeros(50_000, 1))

    m = _strategy_metrics(res, alpha=0.95)

    assert set(m.keys()) == {"mean_pnl", "std", "cvar_95", "cvar_99", "turnover", "total_cost"}
    assert all(isinstance(v, float) for v in m.values())

    assert abs(m["mean_pnl"] - float(pnl.mean())) < 1e-6
    assert abs(m["std"] - float(pnl.std(unbiased=True))) < 1e-6
    assert abs(m["turnover"] - 3.0) < 1e-6
    assert abs(m["total_cost"] - 0.25) < 1e-6
    assert abs(m["cvar_95"] - empirical_cvar(pnl, 0.95)) < 1e-9
    assert abs(m["cvar_99"] - empirical_cvar(pnl, 0.99)) < 1e-9
    # CVaR_99 is a deeper tail than CVaR_95.
    assert m["cvar_99"] > m["cvar_95"]


# --- Task 12.4: evaluate ---

def test_evaluate_returns_expected_keys_single_instrument():
    from deephedge.hedger import Hedger
    from deephedge.train import _n_features
    from deephedge.evaluate import evaluate

    cfg = _tiny_gbm_cfg(model="gbm", cost=0.001, n_paths=2048, batch_size=2048)
    torch.manual_seed(0)
    hedger = Hedger(_n_features(cfg), cfg.n_instruments)

    results = evaluate(cfg, hedger)

    # single-instrument regime -> no bs_delta_vega
    assert set(results.keys()) == {"nn", "bs_delta", "no_hedge"}
    metric_keys = {"mean_pnl", "std", "cvar_95", "cvar_99", "turnover", "total_cost"}
    for name, m in results.items():
        assert set(m.keys()) == metric_keys, name
        assert all(isinstance(v, float) for v in m.values()), name

    # no_hedge never trades -> zero turnover and zero cost.
    assert abs(results["no_hedge"]["turnover"]) < 1e-9
    assert abs(results["no_hedge"]["total_cost"]) < 1e-9
    # under proportional cost, the delta hedger does pay cost.
    assert results["bs_delta"]["total_cost"] > 0.0


def test_evaluate_includes_bs_delta_vega_for_two_instruments():
    from deephedge.hedger import Hedger
    from deephedge.train import _n_features
    from deephedge.evaluate import evaluate

    cfg = _tiny_gbm_cfg(
        model="heston", cost=0.001,
        instruments=("underlying", "option"),
        hedge_option_strike=110.0, hedge_option_maturity=60 / 252,
        n_paths=1024, batch_size=1024,
    )
    torch.manual_seed(0)
    hedger = Hedger(_n_features(cfg), cfg.n_instruments)

    results = evaluate(cfg, hedger)

    assert set(results.keys()) == {"nn", "bs_delta", "no_hedge", "bs_delta_vega"}
    assert all(isinstance(v, float) for v in results["bs_delta_vega"].values())


# --- Task 12.5: metrics_table ---

def test_metrics_table_is_nonempty_markdown():
    from deephedge.evaluate import metrics_table

    results = {
        "nn": {"mean_pnl": 0.01, "std": 0.5, "cvar_95": 1.2, "cvar_99": 1.8,
               "turnover": 3.0, "total_cost": 0.05},
        "bs_delta": {"mean_pnl": -0.02, "std": 0.7, "cvar_95": 1.6, "cvar_99": 2.4,
                     "turnover": 5.0, "total_cost": 0.09},
        "no_hedge": {"mean_pnl": 0.0, "std": 4.0, "cvar_95": 9.0, "cvar_99": 13.0,
                     "turnover": 0.0, "total_cost": 0.0},
    }

    table = metrics_table(results)

    assert isinstance(table, str)
    assert len(table) > 0
    # markdown table structure: header separator row of dashes/pipes.
    lines = table.strip().splitlines()
    assert lines[0].startswith("|")
    assert set(lines[1].replace("|", "").replace(":", "").strip()) <= {"-", " "}
    # one data row per strategy + header + separator.
    assert len(lines) == 2 + len(results)
    # every strategy name appears.
    for name in results:
        assert name in table
    # column headers present.
    for col in ("mean_pnl", "std", "cvar_95", "cvar_99", "turnover", "total_cost"):
        assert col in table
    # values are rendered (4-decimal formatting of cvar_95 for nn).
    assert "1.2000" in table


# --- Task 12.6: plot_pnl_distribution ---

def test_plot_pnl_distribution_writes_file(tmp_path):
    from deephedge.evaluate import plot_pnl_distribution

    gen = torch.Generator().manual_seed(1)
    results = {
        "nn": {"pnl": 0.3 * torch.randn(5000, generator=gen), "cvar_95": 0.6},
        "bs_delta": {"pnl": 0.5 * torch.randn(5000, generator=gen), "cvar_95": 1.0},
        "no_hedge": {"pnl": 3.0 * torch.randn(5000, generator=gen), "cvar_95": 6.0},
    }
    out = tmp_path / "pnl_dist.png"

    ret = plot_pnl_distribution(results, str(out))

    assert out.exists()
    assert out.stat().st_size > 0
    # convention: returns the saved path for chaining in scripts.
    assert ret == str(out)


# --- Task 12.7: plot_hedge_ratio ---

def test_plot_hedge_ratio_writes_file(tmp_path):
    from deephedge.hedger import Hedger
    from deephedge.train import _n_features
    from deephedge.evaluate import plot_hedge_ratio

    cfg = _tiny_gbm_cfg(model="gbm", cost=0.01, n_steps=20)
    torch.manual_seed(0)
    hedger = Hedger(_n_features(cfg), cfg.n_instruments)
    out = tmp_path / "hedge_ratio.png"

    ret = plot_hedge_ratio(cfg, hedger, str(out))

    assert out.exists()
    assert out.stat().st_size > 0
    assert ret == str(out)


def test_plot_hedge_ratio_comment_cites_whalley_wilmott():
    # Spec §12: the band finding must cite Whalley-Wilmott, NOT Leland.
    import inspect

    from deephedge.evaluate import plot_hedge_ratio

    src = inspect.getsource(plot_hedge_ratio)
    assert "Whalley" in src
    assert "Leland" not in src
