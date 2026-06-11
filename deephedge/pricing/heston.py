"""Heston semi-analytic pricer: characteristic function + Carr-Madan FFT pricer.

Spec refs: §7.2, §13(2). The constant part of the drift coefficient is `kappa`,
NOT `kappa - rho*xi` — mixing them mis-prices 5-14%. Albrecher (2007) g2/-d form
with the principal (NumPy) square root keeps the complex log on the right branch
as tau grows.
"""
from __future__ import annotations

from dataclasses import replace as _replace

import numpy as np
import torch

from deephedge.config import ExperimentConfig
from deephedge.instruments import EuropeanOption, payoff
from deephedge.simulators.heston import simulate_heston


def heston_char_func(u, cfg: ExperimentConfig, tau: float) -> np.ndarray:
    """phi(u) = E[exp(i*u*ln S_tau)] under Heston, in NumPy complex.

    Constant drift coefficient is `kappa` (NOT kappa - rho*xi). Principal sqrt.
    `u` may be a real or complex array/scalar; returns a complex128 ndarray.
    """
    u = np.asarray(u, dtype=np.complex128)
    kappa = cfg.kappa
    theta = cfg.theta
    xi = cfg.xi
    rho = cfg.rho
    v0 = cfg.v0
    s0 = cfg.s0
    r = cfg.r
    q = cfg.q

    beta = kappa - rho * xi * 1j * u                      # constant coeff = kappa
    d = np.sqrt(beta**2 + xi**2 * (u**2 + 1j * u))        # principal sqrt, Re(d) >= 0
    g2 = (beta - d) / (beta + d)                          # Albrecher g2/-d form
    edt = np.exp(-d * tau)
    D = ((beta - d) / xi**2) * ((1 - edt) / (1 - g2 * edt))
    C = (kappa * theta / xi**2) * (
        (beta - d) * tau - 2 * np.log((1 - g2 * edt) / (1 - g2))
    )
    phi = np.exp(C + D * v0 + 1j * u * (np.log(s0) + (r - q) * tau))
    return phi


def _cm_call_price(cfg: ExperimentConfig, K: float, tau: float,
                   damping: float, n_grid: int) -> float:
    """Carr-Madan damped-call price via trapezoidal integration over v in (0, v_max]."""
    r = cfg.r
    lnK = np.log(K)
    # Fine grid on (0, v_max]; start just above 0 to avoid the integrand pole at v=0.
    v_max = 200.0
    v = np.linspace(1e-8, v_max, n_grid)
    # psi(v) = exp(-r*tau) * phi(v - (damping+1)i) / (damping^2 + damping - v^2 + i(2*damping+1)v)
    phi = heston_char_func(v - (damping + 1.0) * 1j, cfg, tau)
    denom = damping**2 + damping - v**2 + 1j * (2.0 * damping + 1.0) * v
    psi = np.exp(-r * tau) * phi / denom
    integrand = np.real(np.exp(-1j * v * lnK) * psi)
    integral = np.trapezoid(integrand, v)
    call = np.exp(-damping * lnK) / np.pi * integral
    return float(call)


def heston_price_cm(cfg: ExperimentConfig, K: float, tau: float, kind: str = "call",
                    *, damping: float = 1.5, n_grid: int = 4096) -> float:
    """Heston European price via Carr-Madan damped-call FFT/integral.

    Put is obtained from the call by put-call parity.
    """
    call = _cm_call_price(cfg, K, tau, damping, n_grid)
    if kind == "call":
        return call
    if kind == "put":
        # parity: C - P = s0*exp(-q*tau) - K*exp(-r*tau)
        fwd = cfg.s0 * np.exp(-cfg.q * tau) - K * np.exp(-cfg.r * tau)
        return float(call - fwd)
    raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


def heston_price_mc(cfg: ExperimentConfig, K: float, tau: float, n_paths: int,
                    generator: torch.Generator, kind: str = "call") -> tuple[float, float]:
    """Monte-Carlo Heston price (discounted mean payoff) and standard error.

    Simulates under risk-neutral drift (mu = r) over horizon `tau` using the Heston
    full-truncation Euler simulator, then discounts the terminal European payoff.
    The number of time steps is taken from cfg.n_steps (or uses a minimum of 30 steps
    per year to control discretization bias when tau differs from cfg.maturity).
    """
    # Use at least 30 steps per year scaled to tau for adequate discretization.
    n_steps = max(cfg.n_steps, max(30, int(30 * tau)))
    sim_cfg = _replace(cfg, mu=cfg.r, maturity=tau, n_steps=n_steps)
    paths = simulate_heston(sim_cfg, n_paths, generator)
    S_T = paths.S[:, -1]
    option = EuropeanOption(strike=K, maturity=tau, kind=kind)
    disc = float(np.exp(-cfg.r * tau))
    pay = payoff(option, S_T) * disc           # (n_paths,)
    price = float(pay.mean().item())
    stderr = float((pay.std(unbiased=True) / np.sqrt(n_paths)).item())
    return price, stderr


def _heston_P1(cfg: ExperimentConfig, K: float, tau: float, n_grid: int = 4096) -> float:
    """In-the-money delta probability P1 via Gil-Pelaez under the share measure.

    P1 = 1/2 + (1/pi) * integral_0^inf Re[ exp(-i*v*lnK) * phi(v - i) / (i*v*phi(-i)) ] dv
    where phi(-i) = forward = s0*exp((r-q)*tau).
    """
    lnK = np.log(K)
    v = np.linspace(1e-8, 200.0, n_grid)
    fwd_cf = heston_char_func(np.array([-1j]), cfg, tau)[0]   # phi(-i) = forward
    num = heston_char_func(v - 1j, cfg, tau)
    integrand = np.real(np.exp(-1j * v * lnK) * num / (1j * v * fwd_cf))
    integral = np.trapezoid(integrand, v)
    return float(0.5 + integral / np.pi)


def heston_delta(cfg: ExperimentConfig, K: float, tau: float, kind: str = "call") -> float:
    """Heston European delta = exp(-q*tau) * P1 (call). Put via parity.

    Other greeks (gamma, vega-to-v0/theta/xi, theta) are obtained by central-difference
    bumping of `heston_price_cm` (relative bump ~1e-4); only delta is closed-form here.
    """
    P1 = _heston_P1(cfg, K, tau)
    call_delta = np.exp(-cfg.q * tau) * P1
    if kind == "call":
        return float(call_delta)
    if kind == "put":
        return float(call_delta - np.exp(-cfg.q * tau))   # parity: put_d = call_d - e^{-q*tau}
    raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
