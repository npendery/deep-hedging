# tests/pricing/test_heston.py
import numpy as np
import pytest
import torch

from deephedge.config import ExperimentConfig
from deephedge.pricing.heston import heston_char_func


def _cfg(**kw):
    base = dict(
        s0=100.0, k=100.0, r=0.0, q=0.0,
        v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7,
    )
    base.update(kw)
    return ExperimentConfig(**base)


def test_char_func_phi_at_zero_is_one():
    # phi(0) = E[e^{i*0*lnS_T}] = 1 exactly. Necessary sanity check.
    cfg = _cfg()
    phi0 = heston_char_func(np.array([0.0 + 0.0j]), cfg, tau=1.0)
    assert phi0.shape == (1,)
    assert np.iscomplexobj(phi0)
    np.testing.assert_allclose(phi0[0], 1.0 + 0.0j, atol=1e-12)


def test_char_func_uses_kappa_drift_not_kappa_minus_rhoxi():
    # The C term contains (kappa*theta/xi**2)*[(beta-d)*tau - 2*log(...)].
    # With the CORRECT constant drift coeff = kappa, recompute C+D*v0 by hand at u=1
    # and assert phi matches exp(C + D*v0 + i*u*(ln s0 + (r-q)*tau)).
    cfg = _cfg()
    tau = 0.75
    u = np.array([1.0 + 0.0j])
    kappa, theta, xi, rho, v0 = cfg.kappa, cfg.theta, cfg.xi, cfg.rho, cfg.v0
    s0, r, q = cfg.s0, cfg.r, cfg.q
    beta = kappa - rho * xi * 1j * u
    d = np.sqrt(beta**2 + xi**2 * (u**2 + 1j * u))
    g2 = (beta - d) / (beta + d)
    edt = np.exp(-d * tau)
    D = ((beta - d) / xi**2) * ((1 - edt) / (1 - g2 * edt))
    C = (kappa * theta / xi**2) * ((beta - d) * tau - 2 * np.log((1 - g2 * edt) / (1 - g2)))
    expected = np.exp(C + D * v0 + 1j * u * (np.log(s0) + (r - q) * tau))
    got = heston_char_func(u, cfg, tau)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_char_func_returns_numpy_complex_array_broadcasting():
    cfg = _cfg()
    u = np.linspace(0.0, 50.0, 8)  # real grid -> coerced to complex
    phi = heston_char_func(u, cfg, tau=0.5)
    assert phi.shape == (8,)
    assert phi.dtype == np.complex128
