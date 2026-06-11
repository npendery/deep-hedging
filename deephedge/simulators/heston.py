"""Heston stochastic-volatility simulator (full-truncation Euler).

Dynamics (spec §5.2):
    dS = drift * S dt + sqrt(V) * S dW1
    dV = kappa*(theta - V) dt + xi*sqrt(V) dW2,   corr(dW1, dW2) = rho

Full-truncation Euler (Lord, Koekkoek & van Dijk 2010): truncate ONLY the function
fed to the coefficients via V_plus = max(V_i, 0); the carried state V_{i+1} is NOT
truncated and is allowed to go negative.

    V_plus     = max(V_i, 0)
    V_{i+1}    = V_i + kappa*(theta - V_plus)*dt + xi*sqrt(V_plus)*sqrt(dt)*Z2
    logS_{i+1} = logS_i + (drift - 0.5*V_plus)*dt + sqrt(V_plus)*sqrt(dt)*Z1

    Z1 = Za ;  Z2 = rho*Za + sqrt(1-rho^2)*Zb ;  Za, Zb iid N(0,1)

Feller condition 2*kappa*theta >= xi^2: when satisfied (strict), the *continuous*
variance stays strictly positive. Independently of Feller, the Euler *discretization*
can still produce negative V because the Gaussian increment is unbounded -- which is
the actual reason truncation of the coefficient function is needed (spec §5.2).
"""
import math

import torch

from deephedge.config import ExperimentConfig
from deephedge.simulators.base import Paths


def simulate_heston(
    cfg: ExperimentConfig, n_paths: int, generator: torch.Generator
) -> Paths:
    if cfg.heston_scheme == "qe":
        # Optional Andersen (2008) QE scheme -- not implemented in V1; full
        # truncation is the default. See Andersen (2008), "Simple and efficient
        # simulation of the Heston model" for the quadratic-exponential scheme.
        raise NotImplementedError(
            "heston_scheme='qe' (Andersen 2008 QE) is not implemented; "
            "use heston_scheme='full_truncation' (the V1 default)."
        )

    device = torch.device(cfg.device)
    n_steps = cfg.n_steps
    dt = cfg.dt
    sqrt_dt = math.sqrt(dt)
    drift = cfg.drift

    logS = torch.empty((n_paths, n_steps + 1), device=device)
    V = torch.empty((n_paths, n_steps + 1), device=device)
    logS[:, 0] = math.log(cfg.s0)
    V[:, 0] = cfg.v0

    rho = cfg.rho
    sqrt_one_minus_rho2 = math.sqrt(1.0 - rho * rho)

    for i in range(n_steps):
        za = torch.randn((n_paths,), generator=generator, device=device)
        zb = torch.randn((n_paths,), generator=generator, device=device)
        z1 = za
        z2 = rho * za + sqrt_one_minus_rho2 * zb

        v_plus = V[:, i].clamp(min=0.0)
        sqrt_v_plus = v_plus.sqrt()

        V[:, i + 1] = (
            V[:, i]
            + cfg.kappa * (cfg.theta - v_plus) * dt
            + cfg.xi * sqrt_v_plus * sqrt_dt * z2
        )
        logS[:, i + 1] = (
            logS[:, i]
            + (drift - 0.5 * v_plus) * dt
            + sqrt_v_plus * sqrt_dt * z1
        )

    S = logS.exp()
    times = torch.linspace(0.0, cfg.maturity, n_steps + 1, device=device)
    return Paths(S=S, V=V, dt=dt, times=times)
