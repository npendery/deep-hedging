"""Heston semi-analytic pricer: characteristic function + Carr-Madan FFT pricer.

Spec refs: §7.2, §13(2). The constant part of the drift coefficient is `kappa`,
NOT `kappa - rho*xi` — mixing them mis-prices 5-14%. Albrecher (2007) g2/-d form
with the principal (NumPy) square root keeps the complex log on the right branch
as tau grows.
"""
from __future__ import annotations

import numpy as np

from deephedge.config import ExperimentConfig


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
