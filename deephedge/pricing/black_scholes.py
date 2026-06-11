"""Black-Scholes price and Greeks (torch, broadcasting, safe tau->0).

All functions accept ``tau`` as a Python float or a torch.Tensor and broadcast
over ``S``, ``K``, ``sigma``. The ``tau->0`` limit is regularized by clamping
tau to a small floor inside ``_d1_d2`` so d1/d2 never divide by zero; callers
that need exact intrinsic value at expiry handle the boundary explicitly.
"""
from __future__ import annotations

import torch

# Floor applied to tau so sqrt(tau) and the 1/(sigma*sqrt(tau)) terms stay finite.
_TAU_FLOOR = 1e-12


def _as_tensor(x, ref: torch.Tensor) -> torch.Tensor:
    """Promote a float/int to a tensor on the same device/dtype as ``ref``."""
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x, dtype=ref.dtype, device=ref.device)


def _d1_d2(S, K, tau, r, sigma, q):
    """Return (d1, d2, sqrt_tau) with tau clamped to a small positive floor.

    d1 = (log(S/K) + (r - q + 0.5*sigma**2)*tau) / (sigma*sqrt(tau))
    d2 = d1 - sigma*sqrt(tau)
    """
    S = torch.as_tensor(S, dtype=torch.get_default_dtype())
    K = _as_tensor(K, S)
    sigma = _as_tensor(sigma, S)
    tau = _as_tensor(tau, S)
    r = _as_tensor(r, S)
    q = _as_tensor(q, S)
    tau_safe = torch.clamp(tau, min=_TAU_FLOOR)
    sqrt_tau = torch.sqrt(tau_safe)
    vol_sqrt_tau = sigma * sqrt_tau
    d1 = (torch.log(S / K) + (r - q + 0.5 * sigma * sigma) * tau_safe) / vol_sqrt_tau
    d2 = d1 - vol_sqrt_tau
    return d1, d2, sqrt_tau


def bs_price(S, K, tau, r, sigma, q=0.0, kind="call") -> torch.Tensor:
    """Black-Scholes European option price.

    At/below expiry (tau <= 0) returns exact intrinsic value, broadcast-safe and
    NaN-free. For tau > 0 uses the standard formula with the clamped d1/d2.
    """
    S = torch.as_tensor(S, dtype=torch.get_default_dtype())
    K = _as_tensor(K, S)
    r = _as_tensor(r, S)
    q = _as_tensor(q, S)
    tau = _as_tensor(tau, S)
    d1, d2, _ = _d1_d2(S, K, tau, r, sigma, q)
    disc = torch.exp(-r * tau)
    disc_div = torch.exp(-q * tau)
    if kind == "call":
        bs = S * disc_div * torch.special.ndtr(d1) - K * disc * torch.special.ndtr(d2)
        intrinsic = torch.clamp(S - K, min=0.0)
    elif kind == "put":
        bs = K * disc * torch.special.ndtr(-d2) - S * disc_div * torch.special.ndtr(-d1)
        intrinsic = torch.clamp(K - S, min=0.0)
    else:
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    expired = tau <= 0.0
    return torch.where(torch.broadcast_to(expired, bs.shape), torch.broadcast_to(intrinsic, bs.shape), bs)


def bs_delta(S, K, tau, r, sigma, q=0.0, kind="call") -> torch.Tensor:
    """Black-Scholes delta = exp(-q*tau)*N(d1) for a call, that minus exp(-q*tau) for a put."""
    S = torch.as_tensor(S, dtype=torch.get_default_dtype())
    q = _as_tensor(q, S)
    tau = _as_tensor(tau, S)
    d1, _, _ = _d1_d2(S, K, tau, r, sigma, q)
    disc_div = torch.exp(-q * tau)
    if kind == "call":
        return disc_div * torch.special.ndtr(d1)
    elif kind == "put":
        return disc_div * (torch.special.ndtr(d1) - 1.0)
    raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


_INV_SQRT_2PI = 0.3989422804014327  # 1/sqrt(2*pi)


def _norm_pdf(x: torch.Tensor) -> torch.Tensor:
    """Standard-normal PDF phi(x) = exp(-x^2/2)/sqrt(2*pi)."""
    return _INV_SQRT_2PI * torch.exp(-0.5 * x * x)


def _zero_at_expiry(value: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """Force value to 0 wherever tau<=0 (Greeks vanish at expiry), broadcast-safe."""
    expired = torch.broadcast_to(tau <= 0.0, value.shape)
    return torch.where(expired, torch.zeros_like(value), value)


def bs_vega(S, K, tau, r, sigma, q=0.0) -> torch.Tensor:
    """Black-Scholes vega = S*exp(-q*tau)*phi(d1)*sqrt(tau). Zero at expiry."""
    S = torch.as_tensor(S, dtype=torch.get_default_dtype())
    q = _as_tensor(q, S)
    tau = _as_tensor(tau, S)
    d1, _, sqrt_tau = _d1_d2(S, K, tau, r, sigma, q)
    value = S * torch.exp(-q * tau) * _norm_pdf(d1) * sqrt_tau
    return _zero_at_expiry(value, tau)


def bs_gamma(S, K, tau, r, sigma, q=0.0) -> torch.Tensor:
    """Black-Scholes gamma = exp(-q*tau)*phi(d1)/(S*sigma*sqrt(tau)). Zero at expiry."""
    S = torch.as_tensor(S, dtype=torch.get_default_dtype())
    q = _as_tensor(q, S)
    sigma_t = _as_tensor(sigma, S)
    tau = _as_tensor(tau, S)
    d1, _, sqrt_tau = _d1_d2(S, K, tau, r, sigma, q)
    value = torch.exp(-q * tau) * _norm_pdf(d1) / (S * sigma_t * sqrt_tau)
    return _zero_at_expiry(value, tau)


def bs_theta(S, K, tau, r, sigma, q=0.0, kind="call") -> torch.Tensor:
    """Black-Scholes theta (per-year). Zero at expiry; NaN-free via clamped d1/d2."""
    S = torch.as_tensor(S, dtype=torch.get_default_dtype())
    K = _as_tensor(K, S)
    r = _as_tensor(r, S)
    q = _as_tensor(q, S)
    sigma_t = _as_tensor(sigma, S)
    tau = _as_tensor(tau, S)
    d1, d2, sqrt_tau = _d1_d2(S, K, tau, r, sigma, q)
    disc = torch.exp(-r * tau)
    disc_div = torch.exp(-q * tau)
    gamma_term = -S * disc_div * _norm_pdf(d1) * sigma_t / (2.0 * sqrt_tau)
    if kind == "call":
        value = (
            gamma_term
            - r * K * disc * torch.special.ndtr(d2)
            + q * S * disc_div * torch.special.ndtr(d1)
        )
    elif kind == "put":
        value = (
            gamma_term
            + r * K * disc * torch.special.ndtr(-d2)
            - q * S * disc_div * torch.special.ndtr(-d1)
        )
    else:
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    return _zero_at_expiry(value, tau)
