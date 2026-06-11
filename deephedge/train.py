"""Deep-hedging training loop (Buehler et al. 2019).

Adam over the shared-weight MLP hedger (and, for CVaR, the auxiliary scalar w),
backpropagating through fresh Monte-Carlo trajectories each step (spec §11).
"""
from __future__ import annotations

import torch
from torch import nn

from deephedge.config import ExperimentConfig
from deephedge.hedger import Hedger
from deephedge.instruments import EuropeanOption
from deephedge.pricing.black_scholes import bs_price
from deephedge.pricing.heston import heston_price_cm
from deephedge.portfolio import build_instr_prices, simulate_pnl
from deephedge.benchmarks import make_nn_strategy
from deephedge.losses import cvar_loss, entropic_loss
from deephedge.simulators.base import get_simulator

_STOCH_VOL_MODELS = {"heston", "bates"}


def _n_features(cfg: ExperimentConfig) -> int:
    """Feature width emitted by build_features for this cfg.

    Contract feature order: [log(S_i/k_norm), tau, *prev_holdings, (sqrt(V_i) if not None)]
      -> 2 base features + n_instruments prior holdings + 1 if the model carries variance.
    """
    extra = 1 if cfg.model in _STOCH_VOL_MODELS else 0
    return 2 + cfg.n_instruments + extra


def _premium(cfg: ExperimentConfig, option: EuropeanOption) -> float:
    """Risk-neutral model price of the sold option (spec §3 measure convention).

    Jump models (merton/bates) use the diffusion-only model price as an explicit V1
    approximation: BS for gbm/merton, Heston (heston_price_cm) for heston/bates — i.e.
    jump-model premiums ignore jumps. This is harmless for the headline because the
    identical premium is given to every strategy (a constant additive offset that cancels
    exactly in the relative comparison and in all tail/CVaR differences). See header
    "Premium convention (V1)".
    """
    if cfg.model in _STOCH_VOL_MODELS:
        return float(heston_price_cm(cfg, option.strike, option.maturity, kind=option.kind))
    S = torch.tensor(cfg.s0, dtype=torch.float64, device=cfg.device)
    p = bs_price(S, cfg.k, option.maturity, cfg.r, cfg.sigma, q=cfg.q, kind=option.kind)
    return float(p)


def _hedge_option(cfg: ExperimentConfig) -> EuropeanOption:
    """The HEDGE instrument's option contract, defaulting to the sold call."""
    strike = cfg.hedge_option_strike if cfg.hedge_option_strike is not None else cfg.k
    maturity = cfg.hedge_option_maturity if cfg.hedge_option_maturity is not None else cfg.maturity
    return EuropeanOption(strike, maturity, kind="call")


def train(cfg: ExperimentConfig) -> tuple[Hedger, dict]:
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)          # deterministic MLP weight init
    hedger = Hedger(_n_features(cfg), cfg.n_instruments).to(device)

    params = [{"params": hedger.parameters(), "lr": cfg.lr}]
    w = None
    if cfg.loss == "cvar":
        w = nn.Parameter(torch.zeros((), device=device))
        # separate group with a faster LR for the well-conditioned 1-D w (spec §6.1).
        params.append({"params": [w], "lr": cfg.lr * 10.0})
    opt = torch.optim.Adam(params)

    simulate = get_simulator(cfg.model)
    # liability = the SOLD option (drives payoff/premium); hedge = the marked HEDGE option
    # (instr_prices col1), or None when there is no option leg (single-instrument).
    liability = EuropeanOption(cfg.k, cfg.maturity)
    hedge = _hedge_option(cfg) if "option" in cfg.instruments else None
    premium = _premium(cfg, liability)

    history: dict = {"loss": [], "w": [], "premium": premium}
    total_steps = cfg.epochs * cfg.steps_per_epoch

    for step in range(total_steps):
        gen = torch.Generator(device=device)
        gen.manual_seed(cfg.seed + step)
        paths = simulate(cfg, cfg.batch_size, gen)
        instr_prices = build_instr_prices(cfg, paths, hedge)
        strategy = make_nn_strategy(hedger, cfg)
        pnl = simulate_pnl(strategy, paths, cfg, liability, premium, instr_prices).pnl

        if cfg.loss == "cvar":
            loss = cvar_loss(pnl, cfg.alpha, w)
        else:
            loss = entropic_loss(pnl, cfg.entropic_lambda)

        opt.zero_grad()
        loss.backward()
        opt.step()

        history["loss"].append(float(loss.detach()))
        if w is not None:
            history["w"].append(float(w.detach()))

    return hedger, history
