from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    # contract
    s0: float = 100.0
    k: float = 100.0
    r: float = 0.0
    q: float = 0.0
    mu: float | None = None              # None -> risk-neutral (drift = r)
    maturity: float = 30 / 252           # T (years)
    n_steps: int = 30
    # GBM
    sigma: float = 0.2
    # Heston / Bates
    v0: float = 0.04
    kappa: float = 1.5
    theta: float = 0.04
    xi: float = 0.5
    rho: float = -0.7                    # xi = vol-of-vol
    heston_scheme: str = "full_truncation"   # "full_truncation" | "qe"
    # jumps (Merton / Bates)
    jump_intensity: float = 0.0          # lambda_J per year
    jump_mean: float = -0.1              # mu_J (log space)
    jump_std: float = 0.15               # sigma_J
    # frictions
    cost: float = 0.0                    # proportional transaction-cost rate
    # model + loss
    model: str = "gbm"                   # "gbm" | "heston" | "merton" | "bates"
    loss: str = "cvar"                   # "cvar" | "entropic"
    alpha: float = 0.95                  # CVaR confidence
    entropic_lambda: float = 1.0
    # hedging instruments
    instruments: tuple[str, ...] = ("underlying",)        # or ("underlying", "option")
    hedge_option_strike: float | None = None              # default = k
    hedge_option_maturity: float | None = None            # default = maturity
    # training
    n_paths: int = 100_000
    batch_size: int = 8192
    epochs: int = 50
    steps_per_epoch: int = 50
    lr: float = 1e-3
    seed: int = 0
    device: str = "cpu"
