from dataclasses import dataclass

import torch

from deephedge.pricing.black_scholes import bs_price


@dataclass
class EuropeanOption:
    strike: float
    maturity: float
    kind: str = "call"


def payoff(option: EuropeanOption, S_T) -> torch.Tensor:
    """Terminal European payoff: relu(S_T - K) for a call, relu(K - S_T) for a put."""
    S_T = torch.as_tensor(S_T)
    if option.kind == "call":
        return torch.relu(S_T - option.strike)
    if option.kind == "put":
        return torch.relu(option.strike - S_T)
    raise ValueError(f"Unknown option kind {option.kind!r}; expected 'call' or 'put'")


def mark_option(cfg, paths, option: EuropeanOption) -> torch.Tensor:
    """Mark-to-market price of the hedge `option` at every node of `paths`.

    Returns a (n_paths, n_steps+1) tensor. At step i the time-to-maturity is
    tau_i = max(option.maturity - times_i, 0). For model in {"gbm", "merton"} each node
    is priced with Black-Scholes at sigma=cfg.sigma. For model in {"heston", "bates"} the
    leg is marked with a BS-implied-vol proxy frozen at t0 (V1 approximation; see
    _heston_implied_vol). BS at tau->0 returns intrinsic value, so the terminal column
    equals the option payoff within pricing tolerance.

    The mark is differentiable in paths.S (the autograd graph through the option leg's
    MTM gains stays intact, spec §9).
    """
    S = paths.S                                              # (n_paths, n_steps+1)
    device = S.device
    # tau_i broadcast across paths: (1, n_steps+1), clamped at >= 0.
    tau = (option.maturity - paths.times).clamp(min=0.0).to(device)  # (n_steps+1,)
    tau_row = tau.unsqueeze(0).expand_as(S)                  # (n_paths, n_steps+1)

    sigma = _mark_vol(cfg, option)                           # frozen vol for the leg
    return bs_price(S, option.strike, tau_row, cfg.r, sigma, q=cfg.q, kind=option.kind)


def _mark_vol(cfg, option: EuropeanOption) -> float:
    """Volatility used to mark the option leg.

    For GBM / Merton this is simply cfg.sigma. (Heston / Bates override this in Task
    11.3 with a Heston-implied BS vol; until then they fall back to cfg.sigma.)
    """
    if cfg.model in _STOCH_VOL_MODELS:
        return _heston_implied_vol(cfg, option)
    return cfg.sigma


_STOCH_VOL_MODELS = {"heston", "bates"}


def _heston_implied_vol(cfg, option: EuropeanOption) -> float:
    """Black-Scholes implied vol that reprices the Heston model price of `option` at t0.

    V1 approximation for marking the option leg under Heston/Bates: price the hedge
    option once via Carr-Madan at the config spot/variance, then invert bs_price to a
    single BS vol by bisection. The whole node grid is then marked with bs_price at this
    frozen vol (see mark_option). The implied vol is positive and finite for any
    arbitrage-free Heston price strictly above intrinsic.
    """
    from deephedge.pricing.heston import heston_price_cm  # lazy: avoids import cycle

    target = heston_price_cm(cfg, K=option.strike, tau=option.maturity, kind=option.kind)
    S0 = torch.tensor(cfg.s0)

    def bs(sigma: float) -> float:
        return float(
            bs_price(
                S0, option.strike, option.maturity, cfg.r, sigma, q=cfg.q,
                kind=option.kind,
            )
        )

    lo, hi = 1e-4, 5.0
    # bs_price is monotone increasing in sigma; bracket then bisect.
    if target <= bs(lo):
        return lo
    if target >= bs(hi):
        return hi
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if bs(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-8:
            break
    return 0.5 * (lo + hi)
