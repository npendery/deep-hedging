"""Out-of-sample evaluation + the money-shot figures (spec §12).

Computes terminal-P&L metrics for the learned NN policy vs the BS-delta / no-hedge
(and, in the two-instrument case, BS delta+vega) benchmarks on FRESH out-of-sample
paths, and draws the two headline charts. Loss convention: L = -pnl everywhere (§6).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless / pytest-safe backend; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402

import torch  # noqa: E402

from deephedge.config import ExperimentConfig  # noqa: E402
from deephedge.hedger import build_features  # noqa: E402
from deephedge.instruments import EuropeanOption  # noqa: E402
from deephedge.portfolio import PnLResult, build_instr_prices, simulate_pnl  # noqa: E402
from deephedge.pricing.black_scholes import bs_delta, bs_price  # noqa: E402
from deephedge.pricing.heston import heston_price_cm  # noqa: E402
from deephedge.simulators.base import get_simulator  # noqa: E402
from deephedge.benchmarks import (  # noqa: E402
    make_bs_delta_strategy,
    make_bs_delta_vega_strategy,
    make_nn_strategy,
    make_no_hedge_strategy,
)

_STOCH_VOL_MODELS = {"heston", "bates"}
_EVAL_SEED_OFFSET = 10_000  # disjoint from training's cfg.seed + step draws


def _model_premium(cfg: ExperimentConfig, option: EuropeanOption) -> float:
    """Risk-neutral model price of the sold option (spec §3 measure convention).

    Heston/Bates -> Carr-Madan Heston price; GBM/Merton -> Black-Scholes price.
    Mirrors deephedge.train so train and eval use the identical premium.

    Jump models (merton/bates) use the diffusion-only model price as an explicit V1
    approximation — jump-model premiums ignore jumps. This is harmless because the
    identical premium is given to every strategy (a constant additive offset that cancels
    exactly in the relative comparison and in all tail/CVaR differences). See header
    "Premium convention (V1)".
    """
    if cfg.model in _STOCH_VOL_MODELS:
        return float(heston_price_cm(cfg, option.strike, option.maturity, kind=option.kind))
    S0 = torch.tensor(cfg.s0, dtype=torch.float64, device=cfg.device)
    return float(bs_price(S0, cfg.k, cfg.maturity, cfg.r, cfg.sigma, q=cfg.q, kind=option.kind))


def empirical_cvar(pnl: torch.Tensor, alpha: float) -> float:
    """Empirical CVaR of the loss L = -pnl at confidence `alpha` (spec §6.1).

    Rockafellar-Uryasev objective at its empirical optimum w* = VaR_alpha:
        CVaR_alpha(L) = w* + (1/(1-alpha)) * mean(relu(L - w*)),  w* = quantile(L, alpha).
    Returns a Python float. Equals losses.cvar_loss(pnl, alpha, w*) by construction.
    """
    losses = -pnl.reshape(-1)
    w_star = torch.quantile(losses, alpha)
    tail = torch.relu(losses - w_star).mean()
    cvar = w_star + tail / (1.0 - alpha)
    return float(cvar)


def _strategy_metrics(result: PnLResult, alpha: float = 0.95) -> dict:
    """Flat metric dict for one strategy's PnLResult (spec §12).

    alpha is accepted for API symmetry; cvar_95/cvar_99 always report both tail levels.
    """
    pnl = result.pnl.reshape(-1)
    return {
        "mean_pnl": float(pnl.mean()),
        "std": float(pnl.std(unbiased=True)),
        "cvar_95": empirical_cvar(pnl, 0.95),
        "cvar_99": empirical_cvar(pnl, 0.99),
        "turnover": float(result.turnover.reshape(-1).mean()),
        "total_cost": float(result.cost.reshape(-1).mean()),
    }


def _hedge_option(cfg: ExperimentConfig) -> EuropeanOption:
    """The second (hedging) instrument's option contract, defaulting to the sold call."""
    strike = cfg.hedge_option_strike if cfg.hedge_option_strike is not None else cfg.k
    maturity = cfg.hedge_option_maturity if cfg.hedge_option_maturity is not None else cfg.maturity
    return EuropeanOption(strike, maturity, kind="call")


def evaluate(cfg: ExperimentConfig, hedger) -> dict:
    """Out-of-sample evaluation across strategies (spec §12).

    Fresh paths (seed cfg.seed + 10_000), identical risk-neutral premium for all
    strategies, shared simulate_pnl engine. Returns {strategy_name: metric_dict}.

    Two distinct options thread through (header hedge-option contract): the `liability`
    (sold) option drives the terminal payoff/premium in simulate_pnl, while the `hedge`
    option (its own strike/maturity) is marked into instr_prices col1 AND drives the
    delta+vega benchmark. The SAME `hedge` instance reaches build_instr_prices and the
    vega strategy.
    """
    device = torch.device(cfg.device)
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.seed + _EVAL_SEED_OFFSET)

    simulate = get_simulator(cfg.model)
    paths = simulate(cfg, cfg.n_paths, gen)

    liability = EuropeanOption(cfg.k, cfg.maturity)         # the sold/liability call
    hedge = _hedge_option(cfg) if "option" in cfg.instruments else None
    premium = _model_premium(cfg, liability)
    instr_prices = build_instr_prices(cfg, paths, hedge)

    strategies = {
        "nn": make_nn_strategy(hedger, cfg),
        "bs_delta": make_bs_delta_strategy(cfg),
        "no_hedge": make_no_hedge_strategy(),
    }
    if cfg.n_instruments == 2:
        strategies["bs_delta_vega"] = make_bs_delta_vega_strategy(cfg, hedge)

    results: dict = {}
    with torch.no_grad():
        for name, strat in strategies.items():
            res = simulate_pnl(strat, paths, cfg, liability, premium, instr_prices)
            results[name] = _strategy_metrics(res, cfg.alpha)
    return results


_METRIC_COLUMNS = ("mean_pnl", "std", "cvar_95", "cvar_99", "turnover", "total_cost")


def metrics_table(results: dict) -> str:
    """Markdown table of per-strategy metrics (spec §12 / §15 regime table).

    Rows = strategies (insertion order); columns = the fixed metric set; 4-decimal floats.
    """
    header = "| strategy | " + " | ".join(_METRIC_COLUMNS) + " |"
    sep = "| --- | " + " | ".join(["---"] * len(_METRIC_COLUMNS)) + " |"
    rows = [header, sep]
    for name, metrics in results.items():
        cells = " | ".join(f"{metrics[c]:.4f}" for c in _METRIC_COLUMNS)
        rows.append(f"| {name} | {cells} |")
    return "\n".join(rows)


def plot_pnl_distribution(results: dict, path: str) -> str:
    """Overlaid terminal-P&L histograms with CVaR_95 markers (spec §12 money shot).

    `results` maps strategy -> dict with at least "pnl" (per-path tensor) and "cvar_95"
    (float). A dashed vertical line is drawn at -cvar_95 (the left-tail location in P&L
    space). Saves to `path` and returns it.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, entry in results.items():
        pnl = entry["pnl"].reshape(-1).detach().cpu().numpy()
        ax.hist(pnl, bins=80, density=True, histtype="step", linewidth=1.5, label=name)
        if "cvar_95" in entry:
            ax.axvline(-float(entry["cvar_95"]), linestyle="--", linewidth=1.0,
                       label=f"{name} -CVaR95")
    ax.set_xlabel("terminal hedged P&L")
    ax.set_ylabel("density")
    ax.set_title("Hedged P&L distribution (markers at -CVaR_95)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_hedge_ratio(cfg: ExperimentConfig, hedger, path: str) -> str:
    """Learned underlying holding vs BS delta across moneyness near expiry (spec §12).

    Under proportional cost the learned curve flattens into a no-transaction band around
    delta: rebalance only to the band edge. This is the utility-based / singular-control
    result of Whalley & Wilmott (1997) (band half-width proportional to cost^(1/3)); also
    Hodges-Neuberger (1989), Davis-Panas-Zariphopoulou (1993). It is distinct from a
    periodic adjusted-volatility rehedge.
    """
    device = torch.device(cfg.device)
    tau = max(2.0 * cfg.dt, 1e-3)  # small time-to-maturity, near expiry (years)

    S = torch.linspace(0.7 * cfg.k, 1.3 * cfg.k, 121, device=device).reshape(-1, 1)
    delta = bs_delta(S.reshape(-1), cfg.k, tau, cfg.r, cfg.sigma, q=cfg.q)  # (n,)

    # Feature query: prev_holdings = current BS delta so deviation reveals the band.
    prev_holdings = torch.zeros(S.shape[0], cfg.n_instruments, dtype=S.dtype, device=device)
    prev_holdings[:, 0] = delta
    tau_norm = tau / cfg.maturity  # normalized time-to-maturity feature
    feats = build_features(S, tau_norm, prev_holdings, V_i=None, k_norm=cfg.k)
    with torch.no_grad():
        learned = hedger(feats)[:, 0]  # underlying leg

    moneyness = (S.reshape(-1) / cfg.k).detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(moneyness, delta.detach().cpu().numpy(), label="BS delta", linewidth=1.5)
    ax.plot(moneyness, learned.detach().cpu().numpy(), label="learned holding",
            linewidth=1.5, linestyle="--")
    ax.set_xlabel("moneyness S / K")
    ax.set_ylabel("underlying holding")
    ax.set_title(f"Learned holding vs BS delta near expiry (tau={tau:.4f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
