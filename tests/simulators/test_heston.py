# tests/simulators/test_heston.py
import math
import pytest
import torch

from deephedge.config import ExperimentConfig
from deephedge.simulators.base import Paths
from deephedge.simulators.heston import simulate_heston


def _cfg(**kw) -> ExperimentConfig:
    base = dict(
        s0=100.0, k=100.0, r=0.0, q=0.0, mu=None,
        maturity=1.0, n_steps=50,
        v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7,
        model="heston", device="cpu",
    )
    base.update(kw)
    return ExperimentConfig(**base)


def test_shapes_and_initial_conditions():
    cfg = _cfg(n_steps=50)
    gen = torch.Generator(device="cpu").manual_seed(0)
    n_paths = 1000

    paths = simulate_heston(cfg, n_paths, gen)

    assert isinstance(paths, Paths)
    assert paths.S.shape == (n_paths, cfg.n_steps + 1)
    assert paths.V is not None
    assert paths.V.shape == (n_paths, cfg.n_steps + 1)
    # dt and times metadata
    assert math.isclose(paths.dt, cfg.dt, rel_tol=0, abs_tol=1e-12)
    assert paths.times.shape == (cfg.n_steps + 1,)
    assert math.isclose(paths.times[0].item(), 0.0, abs_tol=1e-12)
    assert math.isclose(paths.times[-1].item(), cfg.maturity, abs_tol=1e-9)
    # initial conditions: every path starts at s0 / v0
    assert torch.allclose(paths.S[:, 0], torch.full((n_paths,), cfg.s0))
    assert torch.allclose(paths.V[:, 0], torch.full((n_paths,), cfg.v0))
    # finite everywhere
    assert torch.isfinite(paths.S).all()
    assert torch.isfinite(paths.V).all()


def test_reproducibility_same_seed_same_paths():
    cfg = _cfg(n_steps=20)
    n_paths = 500

    gen_a = torch.Generator(device="cpu").manual_seed(1234)
    gen_b = torch.Generator(device="cpu").manual_seed(1234)
    paths_a = simulate_heston(cfg, n_paths, gen_a)
    paths_b = simulate_heston(cfg, n_paths, gen_b)

    assert torch.equal(paths_a.S, paths_b.S)
    assert torch.equal(paths_a.V, paths_b.V)

    # A different seed must produce different paths (beyond the fixed t=0 column).
    gen_c = torch.Generator(device="cpu").manual_seed(9999)
    paths_c = simulate_heston(cfg, n_paths, gen_c)
    assert not torch.equal(paths_a.S[:, 1:], paths_c.S[:, 1:])
    assert not torch.equal(paths_a.V[:, 1:], paths_c.V[:, 1:])


def test_discounted_expectation_is_martingale():
    # Under risk-neutral drift = r = 0, the discounted price exp(-r*T)*S_T has
    # mean s0 (spec §13.3). With r = 0 this is simply E[S_T] ≈ s0.
    cfg = _cfg(
        s0=100.0, r=0.0, mu=None,           # mu=None -> drift = r = 0
        maturity=1.0, n_steps=100,
        v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7,
    )
    gen = torch.Generator(device="cpu").manual_seed(7)
    n_paths = 200_000

    paths = simulate_heston(cfg, n_paths, gen)
    s_t = paths.S[:, -1]
    discount = math.exp(-cfg.r * cfg.maturity)
    disc_st = discount * s_t

    mean = disc_st.mean().item()
    stderr = (disc_st.std(unbiased=True) / math.sqrt(n_paths)).item()

    # Feller condition here: 2*kappa*theta = 2*1.5*0.04 = 0.12 >= xi^2 = 0.25 is
    # FALSE (violated), so full-truncation discretization carries a small bias.
    # With 200k paths and 100 steps the discounted mean stays within ~3.5 standard
    # errors of s0; this is the spec §13.3 martingale gate.
    assert abs(mean - cfg.s0) < 3.5 * stderr, (
        f"discounted mean {mean:.4f} not within 3.5*stderr ({3.5 * stderr:.4f}) of "
        f"s0={cfg.s0}"
    )


def test_truncation_function_nonnegative_while_stored_state_may_be_negative():
    # Construct a near-zero-vol, high vol-of-vol regime that strongly VIOLATES the
    # Feller condition (2*kappa*theta >= xi^2) so the Euler increment frequently
    # pushes the *stored* variance below zero. The full-truncation scheme must:
    #   (a) feed only V_plus = max(V_i, 0) (>= 0) into the coefficients, so the
    #       sqrt argument is never negative (no NaNs anywhere), and
    #   (b) STILL allow the carried state V_{i+1} to be negative (it is NOT
    #       truncated before being stored), per spec §5.2.
    cfg = _cfg(
        s0=100.0, r=0.0, mu=None,
        maturity=1.0, n_steps=50,
        v0=1e-4,        # start essentially at zero vol
        kappa=0.1,      # weak mean reversion
        theta=1e-4,     # tiny long-run variance
        xi=1.0,         # large vol-of-vol -> 2*kappa*theta=2e-5 << xi^2=1.0
        rho=-0.7,
    )
    # Feller is violated by construction.
    assert 2 * cfg.kappa * cfg.theta < cfg.xi**2

    gen = torch.Generator(device="cpu").manual_seed(42)
    n_paths = 20_000
    paths = simulate_heston(cfg, n_paths, gen)

    # (a) No NaNs/Infs: proves sqrt was never fed a negative argument -> the
    #     coefficient function V_plus stayed non-negative throughout.
    assert torch.isfinite(paths.S).all()
    assert torch.isfinite(paths.V).all()
    # The truncated function max(V, 0) is non-negative by definition; verify the
    # stored variance, when truncated, is a valid sqrt argument.
    assert (paths.V.clamp(min=0.0) >= 0.0).all()

    # (b) The stored state IS permitted to go negative; in this regime it does.
    #     (If the implementation wrongly clamped the *stored* V, this fails ->
    #      catching the absorption/reflection bug the spec warns against.)
    assert (paths.V < 0.0).any(), (
        "stored variance never went negative in a Feller-violating regime; "
        "the scheme is likely truncating the carried state (wrong family)"
    )


def test_qe_scheme_raises_not_implemented():
    cfg = _cfg(heston_scheme="qe", n_steps=10)
    gen = torch.Generator(device="cpu").manual_seed(0)
    with pytest.raises(NotImplementedError, match="Andersen"):
        simulate_heston(cfg, 100, gen)
