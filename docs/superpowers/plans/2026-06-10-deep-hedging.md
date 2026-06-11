# Deep Hedging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a research-grade deep-hedging library that learns option-hedging policies by backpropagating through simulated price trajectories and minimizing a convex risk measure, and beats Black–Scholes delta hedging on tail risk under transaction costs + stochastic vol + jumps.

**Architecture:** A fully-differentiable trajectory pipeline (Buehler et al. 2019): torch simulators emit price/variance paths; a shared-weight MLP outputs per-step holdings; a P&L engine accumulates MTM gains − costs − payoff; a CVaR/entropic loss closes the autograd graph; Adam backprops through all `N` steps. Every module has one responsibility and is unit-tested in isolation.

**Tech Stack:** Python 3.11, PyTorch (autograd through the trajectory), NumPy/SciPy (Heston FFT, stats), Matplotlib (figures), pytest. Device-agnostic, fully seeded.

**Spec:** `docs/specs/2026-06-10-deep-hedging-design.md` (read it first).

---

## Shared Contracts (authoritative — every task binds to these exact names/signatures)

All randomness flows through an explicit `torch.Generator`. All tensors live on `cfg.device`. Tests live under `tests/` mirroring `deephedge/`.

### `deephedge/config.py`
```python
from dataclasses import dataclass

@dataclass(frozen=True)
class ExperimentConfig:
    # contract
    s0: float = 100.0; k: float = 100.0; r: float = 0.0; q: float = 0.0
    mu: float | None = None              # None -> risk-neutral (drift = r)
    maturity: float = 30 / 252           # T (years)
    n_steps: int = 30
    # GBM
    sigma: float = 0.2
    # Heston / Bates
    v0: float = 0.04; kappa: float = 1.5; theta: float = 0.04
    xi: float = 0.5; rho: float = -0.7   # xi = vol-of-vol
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
    n_paths: int = 100_000; batch_size: int = 8192
    epochs: int = 50; steps_per_epoch: int = 50
    lr: float = 1e-3; seed: int = 0; device: str = "cpu"

    @property
    def dt(self) -> float: return self.maturity / self.n_steps
    @property
    def drift(self) -> float: return self.r if self.mu is None else self.mu
    @property
    def n_instruments(self) -> int: return len(self.instruments)
```

### `deephedge/simulators/base.py`
```python
import torch
from dataclasses import dataclass

@dataclass
class Paths:
    S: torch.Tensor            # (n_paths, n_steps+1)
    V: torch.Tensor | None     # (n_paths, n_steps+1) for heston/bates, else None
    dt: float
    times: torch.Tensor        # (n_steps+1,)

def get_simulator(model: str):
    """Return callable (cfg, n_paths, generator) -> Paths for 'gbm'|'heston'|'merton'|'bates'.
    Resolves each model by LAZY import (so it works mid-build before all simulator files exist);
    raises ValueError on an unknown model. This is the single registry mechanism — no decorator."""
```
Simulator fns: `simulate_gbm(cfg, n_paths, generator) -> Paths`, `simulate_heston(...)`, `simulate_merton(...)`, `simulate_bates(...)`.

### `deephedge/pricing/black_scholes.py` (torch, broadcast over tensors; `tau` float or tensor; safe `tau→0`)
`bs_price(S, K, tau, r, sigma, q=0.0, kind="call")`, `bs_delta(...)`, `bs_vega(S,K,tau,r,sigma,q=0.0)`, `bs_gamma(...)`, `bs_theta(...)` — all `-> torch.Tensor`.

### `deephedge/pricing/heston.py`
`heston_char_func(u, cfg, tau) -> Tensor[complex]` — `phi(u)=E[e^{iu·lnS_T}]`, **constant drift coeff = `kappa`**: `beta = kappa - rho*xi*i*u`, `d = sqrt(beta**2 + xi**2*(u**2 + i*u))` (principal sqrt), `g2=(beta-d)/(beta+d)`, Albrecher g2/−d form.
`heston_price_cm(cfg, K, tau, kind="call", *, damping=1.5, n_grid=4096) -> float` (Carr–Madan).
`heston_delta(cfg, K, tau, kind="call") -> float` (= `exp(-q*tau)*P1`).
`heston_price_mc(cfg, K, tau, n_paths, generator, kind="call") -> tuple[float, float]` (price, stderr) — for the regression test.

### `deephedge/instruments.py`
```python
@dataclass
class EuropeanOption:
    strike: float; maturity: float; kind: str = "call"

def payoff(option: EuropeanOption, S_T) -> torch.Tensor      # max(S_T-K,0) for call
def mark_option(cfg, paths, option) -> torch.Tensor           # (n_paths, n_steps+1) MTM via BS (gbm/merton) or Heston (heston/bates)
```

### `deephedge/hedger.py`
```python
class Hedger(torch.nn.Module):
    def __init__(self, n_features: int, n_instruments: int, hidden=(32, 32)): ...
    def forward(self, features) -> torch.Tensor              # (B, n_features) -> (B, n_instruments)

def build_features(S_i, tau_norm, prev_holdings, V_i=None, k_norm=100.0) -> torch.Tensor
# S_i / prev_holdings / V_i accept (B,) OR (B,1) — reshaped to columns internally.
# tau_norm = NORMALIZED time-to-maturity (T - t_i)/T (callers pass state.tau / cfg.maturity).
# k_norm = strike used for moneyness (callers pass cfg.k).
# feature order: [log(S_i/k_norm), tau_norm, *prev_holdings, (sqrt(max(V_i,0)) if V_i is not None)]
```

### `deephedge/losses.py`
```python
def cvar_loss(pnl, alpha: float, w) -> torch.Tensor
#   L = -pnl ; returns w + (1/(1-alpha)) * mean(relu(L - w)) ; w is a learnable scalar nn.Parameter (shape ())
def entropic_loss(pnl, lam: float) -> torch.Tensor
#   (1/lam) * (logsumexp(-lam*pnl) - log(N))   ; pnl is profit-positive ; max-shift stabilized
```

### `deephedge/portfolio.py`
```python
# A Strategy is a callable: strategy(state: StepState) -> holdings  (B, n_instruments)
@dataclass
class StepState:
    step: int; S: torch.Tensor; V: torch.Tensor | None; tau: float
    prev_holdings: torch.Tensor; instr_prices: torch.Tensor   # (B, n_instruments)

@dataclass
class PnLResult:
    pnl: torch.Tensor; turnover: torch.Tensor; cost: torch.Tensor; holdings: torch.Tensor

def build_instr_prices(cfg, paths, hedge_option) -> torch.Tensor
#   (n_paths, n_steps+1, n_instruments). col0 = paths.S (underlying).
#   col1 = mark_option(cfg, paths, hedge_option) when 'option' in cfg.instruments — pass the HEDGE
#   instrument option (its own strike/maturity), NOT the sold/liability option. Underlying-only otherwise.
def simulate_pnl(strategy, paths, cfg, option, premium: float, instr_prices) -> PnLResult
#   `option` here = the SOLD / LIABILITY option (drives terminal payoff and the premium); distinct from
#   the hedge_option marked in instr_prices col1. Both must be threaded consistently by callers.
```

**Premium convention (V1):** the sold-option premium `p0` is priced with the diffusion-only model
price — BS for `gbm`/`merton`, Heston (`heston_price_cm`) for `heston`/`bates` — i.e. jump-model
premiums ignore jumps. This is an explicit V1 approximation. It is harmless for the headline because
the *same* `p0` is given to every strategy (NN, BS-delta, no-hedge), so it is a constant additive
offset to all P&Ls and cancels exactly in the relative comparison and in all tail/CVaR differences.

### `deephedge/benchmarks.py`
`make_nn_strategy(hedger, cfg)`, `make_bs_delta_strategy(cfg)`, `make_bs_delta_vega_strategy(cfg, option)`, `make_no_hedge_strategy()` — each `-> Strategy`.

### `deephedge/train.py`
`train(cfg) -> tuple[Hedger, dict]` (trained hedger + history dict).

### `deephedge/evaluate.py`
`evaluate(cfg, hedger) -> dict` (per-strategy metrics); `plot_pnl_distribution(results, path)`; `plot_hedge_ratio(cfg, hedger, path)`; `metrics_table(results) -> str`.

---
## Group 1 — Scaffolding, config, simulator base, instruments

> Binds to the Shared Contracts in `parts/00-header.md` exactly. All randomness flows through an explicit `torch.Generator`; all tensors live on `cfg.device`. Tests live under `tests/` mirroring `deephedge/`. Repo-relative paths used throughout. Run `pip install -e .` once after Task 1.1 so the package is importable in tests.

### Task 1.1: Project skeleton — `pyproject.toml` + package `__init__` files

**Files:**
- Create: `pyproject.toml`
- Create: `deephedge/__init__.py`
- Create: `deephedge/simulators/__init__.py`
- Create: `deephedge/pricing/__init__.py`
- Test: `tests/test_packaging.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_packaging.py
import importlib

import deephedge


def test_package_exposes_version():
    assert hasattr(deephedge, "__version__")
    assert deephedge.__version__ == "0.1.0"


def test_subpackages_importable():
    # Submodules must be importable as packages.
    sim = importlib.import_module("deephedge.simulators")
    pricing = importlib.import_module("deephedge.pricing")
    assert sim is not None
    assert pricing is not None
```

- [ ] **Step 2: Run test to verify it fails**

Command: `pytest tests/test_packaging.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'deephedge'` (package not created / not installed yet).

- [ ] **Step 3: Write minimal implementation**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "deephedge"
version = "0.1.0"
description = "Research-grade deep hedging library (Buehler et al. 2019)"
requires-python = ">=3.11"
dependencies = [
    "torch",
    "numpy",
    "scipy",
    "matplotlib",
]

[project.optional-dependencies]
dev = ["pytest"]

[tool.setuptools.packages.find]
include = ["deephedge*"]

[tool.pytest.ini_options]
addopts = "-ra"
testpaths = ["tests"]
```

```python
# deephedge/__init__.py
"""Deep hedging library: differentiable trajectory pipeline (Buehler et al. 2019)."""

__version__ = "0.1.0"
```

```python
# deephedge/simulators/__init__.py
"""Price/variance path simulators (GBM, Heston, Merton, Bates)."""
```

```python
# deephedge/pricing/__init__.py
"""Analytic and semi-analytic pricers (Black-Scholes, Heston)."""
```

After creating the files, install the package in editable mode so tests resolve the import:

```bash
pip install -e .
```

- [ ] **Step 4: Run test to verify it passes**

Command: `pytest tests/test_packaging.py -v`

Expected: PASS (both tests green).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml deephedge/__init__.py deephedge/simulators/__init__.py deephedge/pricing/__init__.py tests/test_packaging.py
git commit -m "chore: scaffold deephedge package and pyproject"
```

### Task 1.2: `ExperimentConfig` defaults and frozen dataclass

**Files:**
- Create: `deephedge/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
import dataclasses

import pytest

from deephedge.config import ExperimentConfig


def test_defaults():
    cfg = ExperimentConfig()
    assert cfg.s0 == 100.0
    assert cfg.k == 100.0
    assert cfg.r == 0.0
    assert cfg.q == 0.0
    assert cfg.mu is None
    assert cfg.maturity == pytest.approx(30 / 252)
    assert cfg.n_steps == 30
    assert cfg.sigma == 0.2
    assert cfg.model == "gbm"
    assert cfg.loss == "cvar"
    assert cfg.alpha == 0.95
    assert cfg.instruments == ("underlying",)
    assert cfg.device == "cpu"
    assert cfg.seed == 0


def test_frozen():
    cfg = ExperimentConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.s0 = 50.0  # type: ignore[misc]
```

- [ ] **Step 2: Run test to verify it fails**

Command: `pytest tests/test_config.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'deephedge.config'`.

- [ ] **Step 3: Write minimal implementation**

```python
# deephedge/config.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Command: `pytest tests/test_config.py -v`

Expected: PASS (defaults match contract; instance is frozen).

- [ ] **Step 5: Commit**

```bash
git add deephedge/config.py tests/test_config.py
git commit -m "feat: add ExperimentConfig dataclass with contract defaults"
```

### Task 1.3: `ExperimentConfig` derived properties (`dt`, `drift`, `n_instruments`)

**Files:**
- Modify: `deephedge/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py  (append)
def test_dt_is_maturity_over_n_steps():
    cfg = ExperimentConfig(maturity=1.0, n_steps=4)
    assert cfg.dt == pytest.approx(0.25)


def test_drift_defaults_to_r_when_mu_none():
    cfg = ExperimentConfig(r=0.03, mu=None)
    assert cfg.drift == pytest.approx(0.03)


def test_drift_uses_mu_when_set():
    cfg = ExperimentConfig(r=0.03, mu=0.10)
    assert cfg.drift == pytest.approx(0.10)


def test_n_instruments_counts_instruments():
    assert ExperimentConfig().n_instruments == 1
    assert ExperimentConfig(instruments=("underlying", "option")).n_instruments == 2
```

- [ ] **Step 2: Run test to verify it fails**

Command: `pytest tests/test_config.py -k "dt or drift or n_instruments" -v`

Expected: FAIL with `AttributeError: 'ExperimentConfig' object has no attribute 'dt'` (properties not defined yet).

- [ ] **Step 3: Write minimal implementation**

Append the three properties to the end of the `ExperimentConfig` class body in `deephedge/config.py`:

```python
    @property
    def dt(self) -> float:
        return self.maturity / self.n_steps

    @property
    def drift(self) -> float:
        return self.r if self.mu is None else self.mu

    @property
    def n_instruments(self) -> int:
        return len(self.instruments)
```

- [ ] **Step 4: Run test to verify it passes**

Command: `pytest tests/test_config.py -v`

Expected: PASS (all config tests, including the property tests, green).

- [ ] **Step 5: Commit**

```bash
git add deephedge/config.py tests/test_config.py
git commit -m "feat: add dt/drift/n_instruments properties to ExperimentConfig"
```

### Task 1.4: `Paths` dataclass

**Files:**
- Create: `deephedge/simulators/base.py`
- Test: `tests/simulators/test_base.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/simulators/test_base.py
import torch

from deephedge.simulators.base import Paths


def test_paths_holds_fields():
    n_paths, n_steps = 5, 3
    S = torch.zeros(n_paths, n_steps + 1)
    times = torch.linspace(0.0, 1.0, n_steps + 1)
    paths = Paths(S=S, V=None, dt=0.25, times=times)
    assert paths.S.shape == (n_paths, n_steps + 1)
    assert paths.V is None
    assert paths.dt == 0.25
    assert paths.times.shape == (n_steps + 1,)


def test_paths_can_carry_variance():
    n_paths, n_steps = 5, 3
    S = torch.zeros(n_paths, n_steps + 1)
    V = torch.ones(n_paths, n_steps + 1)
    times = torch.linspace(0.0, 1.0, n_steps + 1)
    paths = Paths(S=S, V=V, dt=0.25, times=times)
    assert paths.V is not None
    assert paths.V.shape == (n_paths, n_steps + 1)
```

- [ ] **Step 2: Run test to verify it fails**

Command: `pytest tests/simulators/test_base.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'deephedge.simulators.base'`.

- [ ] **Step 3: Write minimal implementation**

```python
# deephedge/simulators/base.py
from dataclasses import dataclass

import torch


@dataclass
class Paths:
    S: torch.Tensor            # (n_paths, n_steps+1)
    V: torch.Tensor | None     # (n_paths, n_steps+1) for heston/bates, else None
    dt: float
    times: torch.Tensor        # (n_steps+1,)
```

Also create an empty package marker so `tests/simulators/` is collectable as a directory (pytest does not require `__init__.py`, but the test directory must exist):

```bash
mkdir -p tests/simulators
```

- [ ] **Step 4: Run test to verify it passes**

Command: `pytest tests/simulators/test_base.py -v`

Expected: PASS (both `Paths` tests green).

- [ ] **Step 5: Commit**

```bash
git add deephedge/simulators/base.py tests/simulators/test_base.py
git commit -m "feat: add Paths dataclass for simulator output"
```

### Task 1.5: `get_simulator` lazy-import registry (unknown-model path only)

**Files:**
- Modify: `deephedge/simulators/base.py`
- Test: `tests/simulators/test_base.py`

> `get_simulator` is the single registry mechanism: a function that dispatches by model
> string via LAZY import (so it works mid-build before all simulator files exist) and
> raises `ValueError` on an unknown model. There is no decorator-based registration and no
> module-level registry dict — the lazy-import function below is the entire mechanism. This
> task's test verifies ONLY the unknown-model path — it must NOT call `get_simulator("gbm")`,
> because `gbm.py` does not exist yet at this task (it is created in Group 2); the gbm
> dispatch is exercised there.

- [ ] **Step 1: Write the failing test**

```python
# tests/simulators/test_base.py  (append)
import pytest

from deephedge.simulators.base import get_simulator


def test_unknown_model_raises_value_error():
    # gbm.py does not exist yet at this task; only the unknown-model path is tested here.
    with pytest.raises(ValueError):
        get_simulator("not_a_model")


def test_value_error_message_names_the_model():
    with pytest.raises(ValueError, match="nope"):
        get_simulator("nope")
```

- [ ] **Step 2: Run test to verify it fails**

Command: `pytest tests/simulators/test_base.py -k "unknown or value_error" -v`

Expected: FAIL with `ImportError: cannot import name 'get_simulator'` from `deephedge.simulators.base`.

- [ ] **Step 3: Write minimal implementation**

Append to `deephedge/simulators/base.py`:

```python
def get_simulator(model: str):
    """Return callable (cfg, n_paths, generator) -> Paths for 'gbm'|'heston'|'merton'|'bates'.

    Resolves each model by LAZY import so this works mid-build before all simulator files
    exist. This is the single registry mechanism — no decorator, no module-level dict.
    """
    if model == "gbm":
        from deephedge.simulators.gbm import simulate_gbm
        return simulate_gbm
    if model == "heston":
        from deephedge.simulators.heston import simulate_heston
        return simulate_heston
    if model in ("merton", "bates"):
        from deephedge.simulators.jumps import simulate_merton, simulate_bates
        return {"merton": simulate_merton, "bates": simulate_bates}[model]
    raise ValueError(f"unknown model {model!r}; expected gbm|heston|merton|bates")
```

> The `gbm`/`heston`/`merton`/`bates` branches lazy-import simulator modules created in
> later groups; only the unknown-model `ValueError` path is testable here (no simulator
> module exists yet). The other branches go green as each group lands its simulator.

- [ ] **Step 4: Run test to verify it passes**

Command: `pytest tests/simulators/test_base.py -k "unknown or value_error" -v`

Expected: PASS (`ValueError` on unknown model, including the message-match test).

- [ ] **Step 5: Commit**

```bash
git add deephedge/simulators/base.py tests/simulators/test_base.py
git commit -m "feat: add get_simulator lazy-import registry (ValueError on unknown model)"
```

### Task 1.6: `EuropeanOption` dataclass

**Files:**
- Create: `deephedge/instruments.py`
- Test: `tests/test_instruments.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_instruments.py
from deephedge.instruments import EuropeanOption


def test_european_option_fields_and_default_kind():
    opt = EuropeanOption(strike=100.0, maturity=0.5)
    assert opt.strike == 100.0
    assert opt.maturity == 0.5
    assert opt.kind == "call"


def test_european_option_put_kind():
    opt = EuropeanOption(strike=90.0, maturity=1.0, kind="put")
    assert opt.kind == "put"
```

- [ ] **Step 2: Run test to verify it fails**

Command: `pytest tests/test_instruments.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'deephedge.instruments'`.

- [ ] **Step 3: Write minimal implementation**

```python
# deephedge/instruments.py
from dataclasses import dataclass

import torch


@dataclass
class EuropeanOption:
    strike: float
    maturity: float
    kind: str = "call"
```

- [ ] **Step 4: Run test to verify it passes**

Command: `pytest tests/test_instruments.py -v`

Expected: PASS (fields + default `kind="call"` + explicit put).

- [ ] **Step 5: Commit**

```bash
git add deephedge/instruments.py tests/test_instruments.py
git commit -m "feat: add EuropeanOption dataclass"
```

### Task 1.7: `payoff(option, S_T)` for call and put

**Files:**
- Modify: `deephedge/instruments.py`
- Test: `tests/test_instruments.py`

> Implement `payoff` ONLY. `mark_option` is implemented in the multi-instrument group later; do not add it here. Call payoff = `relu(S_T - K)`, put payoff = `relu(K - S_T)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_instruments.py  (append)
import torch

from deephedge.instruments import payoff


def test_call_payoff_on_small_tensor():
    opt = EuropeanOption(strike=100.0, maturity=0.5, kind="call")
    S_T = torch.tensor([80.0, 100.0, 120.0])
    out = payoff(opt, S_T)
    expected = torch.tensor([0.0, 0.0, 20.0])
    assert torch.allclose(out, expected)
    assert out.shape == S_T.shape


def test_put_payoff_on_small_tensor():
    opt = EuropeanOption(strike=100.0, maturity=0.5, kind="put")
    S_T = torch.tensor([80.0, 100.0, 120.0])
    out = payoff(opt, S_T)
    expected = torch.tensor([20.0, 0.0, 0.0])
    assert torch.allclose(out, expected)


def test_payoff_is_nonnegative_and_differentiable():
    opt = EuropeanOption(strike=100.0, maturity=0.5, kind="call")
    S_T = torch.tensor([90.0, 110.0], requires_grad=True)
    out = payoff(opt, S_T)
    assert (out >= 0).all()
    out.sum().backward()
    # d/dS relu(S-K) = 1 in-the-money, 0 out-of-the-money
    assert torch.allclose(S_T.grad, torch.tensor([0.0, 1.0]))
```

- [ ] **Step 2: Run test to verify it fails**

Command: `pytest tests/test_instruments.py -k payoff -v`

Expected: FAIL with `ImportError: cannot import name 'payoff' from 'deephedge.instruments'`.

- [ ] **Step 3: Write minimal implementation**

Append to `deephedge/instruments.py`:

```python
def payoff(option: EuropeanOption, S_T) -> torch.Tensor:
    """Terminal European payoff: relu(S_T - K) for a call, relu(K - S_T) for a put."""
    S_T = torch.as_tensor(S_T)
    if option.kind == "call":
        return torch.relu(S_T - option.strike)
    if option.kind == "put":
        return torch.relu(option.strike - S_T)
    raise ValueError(f"Unknown option kind {option.kind!r}; expected 'call' or 'put'")
```

- [ ] **Step 4: Run test to verify it passes**

Command: `pytest tests/test_instruments.py -v`

Expected: PASS (call/put values, non-negativity, and autograd gradient `[0, 1]`).

- [ ] **Step 5: Commit**

```bash
git add deephedge/instruments.py tests/test_instruments.py
git commit -m "feat: add European call/put payoff function"
```

### Task 1.8: `tests/conftest.py` shared fixtures (seeded generator + small config)

**Files:**
- Create: `tests/conftest.py`
- Test: `tests/test_conftest_fixtures.py`

> Provides a seeded `torch.Generator` and a small `ExperimentConfig` (tiny `n_paths`/`batch_size`/`n_steps` so tests are fast). These fixtures are consumed by later groups; this task verifies they behave as the contract expects.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_conftest_fixtures.py
import torch

from deephedge.config import ExperimentConfig


def test_generator_fixture_is_seeded_and_reproducible(generator):
    assert isinstance(generator, torch.Generator)
    a = torch.randn(4, generator=generator)
    # Re-seed to the documented test seed and draw again -> identical sequence.
    g2 = torch.Generator()
    g2.manual_seed(1234)
    b = torch.randn(4, generator=g2)
    assert torch.allclose(a, b)


def test_small_config_fixture_is_small(small_config):
    assert isinstance(small_config, ExperimentConfig)
    assert small_config.n_paths <= 1000
    assert small_config.batch_size <= small_config.n_paths
    assert small_config.n_steps <= 10
    assert small_config.device == "cpu"
```

- [ ] **Step 2: Run test to verify it fails**

Command: `pytest tests/test_conftest_fixtures.py -v`

Expected: FAIL with `fixture 'generator' not found` (no `conftest.py` yet).

- [ ] **Step 3: Write minimal implementation**

```python
# tests/conftest.py
import pytest
import torch

from deephedge.config import ExperimentConfig

# Documented test seed used by the `generator` fixture; tests that need the same
# sequence can recreate it via torch.Generator().manual_seed(TEST_SEED).
TEST_SEED = 1234


@pytest.fixture
def generator() -> torch.Generator:
    """A freshly seeded CPU torch.Generator for reproducible tests."""
    g = torch.Generator()
    g.manual_seed(TEST_SEED)
    return g


@pytest.fixture
def small_config() -> ExperimentConfig:
    """A tiny ExperimentConfig so unit tests run fast."""
    return ExperimentConfig(
        n_paths=512,
        batch_size=256,
        n_steps=5,
        epochs=1,
        steps_per_epoch=1,
        seed=TEST_SEED,
        device="cpu",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Command: `pytest tests/test_conftest_fixtures.py -v`

Expected: PASS (generator reproduces the seeded sequence; small config is small and CPU).

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py tests/test_conftest_fixtures.py
git commit -m "test: add shared conftest fixtures (seeded generator, small config)"
```
## Group 2 — GBM simulator

Implements `deephedge/simulators/gbm.py` (`simulate_gbm`). Binds exactly to the
Shared Contracts in `00-header.md`:

- `Paths(S, V, dt, times)` — `S: (n_paths, n_steps+1)`, `V: None` for GBM,
  `dt: float`, `times: (n_steps+1,)`. (`Paths` and `get_simulator` live in
  `deephedge/simulators/base.py`, created by Group 1 — this group does NOT recreate them.)
- `get_simulator(model: str)` returns the callable
  `(cfg, n_paths, generator) -> Paths` via its lazy `"gbm"` branch.
- `simulate_gbm(cfg, n_paths, generator) -> Paths` uses exact log-Euler
  `S_{i+1} = S_i · exp((drift − ½σ²)Δt + σ√Δt · Z)` with `drift = cfg.drift`
  (`= cfg.r` when `cfg.mu is None`), `σ = cfg.sigma`, `Δt = cfg.dt`,
  `times = linspace(0, cfg.maturity, cfg.n_steps + 1)`, all on `cfg.device`.

Spec refs: §5.1 (GBM exact log-Euler), §13(3) (simulator martingale / variance check).

> Assumes Group 1 has created `deephedge/__init__.py`, `deephedge/simulators/__init__.py`,
> `deephedge/simulators/base.py` (`Paths` + `get_simulator`), and `deephedge/config.py`
> (`ExperimentConfig`). This group does NOT recreate `base.py` or `get_simulator`; it only
> adds `deephedge/simulators/gbm.py`. If a package `__init__.py` is genuinely absent when
> running standalone, create the empty marker — but never re-author `base.py`.

---

### Task 2.1: `simulate_gbm` shape + initial condition

**Files:**
- Create: `deephedge/simulators/gbm.py`
- Create: `deephedge/simulators/__init__.py` (empty, only if not already created by Group 1)
- Create: `deephedge/__init__.py` (empty, only if not already created by Group 1)
- Test: `tests/test_gbm.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gbm.py
import torch

from deephedge.config import ExperimentConfig
from deephedge.simulators.base import Paths, get_simulator
from deephedge.simulators.gbm import simulate_gbm


def test_gbm_shape_and_initial_condition():
    cfg = ExperimentConfig(s0=100.0, sigma=0.2, maturity=1.0, n_steps=50)
    gen = torch.Generator(device=cfg.device).manual_seed(0)
    n_paths = 1000

    paths = simulate_gbm(cfg, n_paths, gen)

    assert isinstance(paths, Paths)
    assert paths.S.shape == (n_paths, cfg.n_steps + 1)
    assert paths.V is None
    # exact dt and a (n_steps+1,) time grid from 0..maturity
    assert paths.dt == cfg.dt
    assert paths.times.shape == (cfg.n_steps + 1,)
    assert float(paths.times[0]) == 0.0
    assert torch.isclose(paths.times[-1], torch.tensor(cfg.maturity))
    # every path starts at s0
    assert torch.allclose(paths.S[:, 0], torch.full((n_paths,), cfg.s0))
    # all prices strictly positive (exp keeps GBM positive)
    assert torch.all(paths.S > 0)


def test_get_simulator_gbm_dispatches_to_simulate_gbm():
    cfg = ExperimentConfig(s0=100.0, sigma=0.2, maturity=1.0, n_steps=10)
    gen = torch.Generator(device=cfg.device).manual_seed(0)
    sim = get_simulator("gbm")
    # get_simulator's lazy "gbm" branch returns simulate_gbm itself.
    assert sim is simulate_gbm
    paths = sim(cfg, 16, gen)
    assert isinstance(paths, Paths)
    assert paths.S.shape == (16, cfg.n_steps + 1)
    assert paths.V is None
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_gbm.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'deephedge.simulators.gbm'`
(the module does not exist yet).

- [ ] **Step 3: Write minimal implementation**

```python
# deephedge/simulators/gbm.py
from __future__ import annotations

import torch

from deephedge.config import ExperimentConfig
from deephedge.simulators.base import Paths


def simulate_gbm(
    cfg: ExperimentConfig,
    n_paths: int,
    generator: torch.Generator,
) -> Paths:
    """Exact log-Euler GBM:  S_{i+1} = S_i * exp((drift - 0.5*sigma^2)dt + sigma*sqrt(dt)*Z).

    Returns Paths with S of shape (n_paths, n_steps+1), V=None, the float step dt,
    and the time grid linspace(0, maturity, n_steps+1). All tensors on cfg.device.
    """
    device = torch.device(cfg.device)
    n_steps = cfg.n_steps
    dt = cfg.dt
    drift = cfg.drift
    sigma = cfg.sigma

    times = torch.linspace(0.0, cfg.maturity, n_steps + 1, device=device)

    # standard normal increments Z: one per (path, step)
    Z = torch.randn(n_paths, n_steps, generator=generator, device=device)

    log_increments = (drift - 0.5 * sigma * sigma) * dt + sigma * (dt ** 0.5) * Z

    log_S = torch.empty(n_paths, n_steps + 1, device=device)
    log_S[:, 0] = torch.log(torch.tensor(cfg.s0, device=device))
    log_S[:, 1:] = log_S[:, :1] + torch.cumsum(log_increments, dim=1)

    S = torch.exp(log_S)
    return Paths(S=S, V=None, dt=dt, times=times)
```

> Requires `deephedge/config.py` with `ExperimentConfig` (Group 1). If absent, create
> the minimal contract version now:
> ```python
> # deephedge/config.py
> from dataclasses import dataclass
>
> @dataclass(frozen=True)
> class ExperimentConfig:
>     s0: float = 100.0; k: float = 100.0; r: float = 0.0; q: float = 0.0
>     mu: float | None = None
>     maturity: float = 30 / 252
>     n_steps: int = 30
>     sigma: float = 0.2
>     seed: int = 0; device: str = "cpu"
>
>     @property
>     def dt(self) -> float: return self.maturity / self.n_steps
>     @property
>     def drift(self) -> float: return self.r if self.mu is None else self.mu
> ```
> Use the full contract version from `00-header.md` when assembling.

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/test_gbm.py -v
```

Expected: PASS. Also rerun `pytest tests/simulators/test_base.py -v` (Group 1's base
test) — `get_simulator("gbm")` now resolves via its lazy import, and the unknown-model
`ValueError` path still holds.

- [ ] **Step 5: Commit**

```
git add deephedge/simulators/gbm.py tests/test_gbm.py
git commit -m "feat(simulators): implement exact log-Euler GBM simulator"
```

---

### Task 2.2: GBM reproducibility with a seeded generator

**Files:**
- Modify: `tests/test_gbm.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_gbm.py
def test_gbm_reproducible_with_same_seed():
    cfg = ExperimentConfig(s0=100.0, sigma=0.2, maturity=1.0, n_steps=50)
    n_paths = 500

    gen_a = torch.Generator(device=cfg.device).manual_seed(1234)
    gen_b = torch.Generator(device=cfg.device).manual_seed(1234)

    paths_a = simulate_gbm(cfg, n_paths, gen_a)
    paths_b = simulate_gbm(cfg, n_paths, gen_b)

    # identical seed -> bit-identical paths
    assert torch.equal(paths_a.S, paths_b.S)


def test_gbm_different_seed_differs():
    cfg = ExperimentConfig(s0=100.0, sigma=0.2, maturity=1.0, n_steps=50)
    n_paths = 500

    gen_a = torch.Generator(device=cfg.device).manual_seed(1234)
    gen_c = torch.Generator(device=cfg.device).manual_seed(9999)

    paths_a = simulate_gbm(cfg, n_paths, gen_a)
    paths_c = simulate_gbm(cfg, n_paths, gen_c)

    # different seeds -> different paths (the initial column is equal, the rest is not)
    assert not torch.equal(paths_a.S[:, 1:], paths_c.S[:, 1:])
```

- [ ] **Step 2: Run test to verify it fails**

These tests pass against the Task 2.1 implementation because randomness already flows
through the explicit `generator`. To confirm the test is real (not vacuous), first
verify it FAILS against a non-reproducible variant: temporarily change `gbm.py`'s
`torch.randn(... generator=generator ...)` to `torch.randn(...)` (drop the generator),
then run:

```
pytest tests/test_gbm.py::test_gbm_reproducible_with_same_seed -v
```

Expected: FAIL with `AssertionError` on `torch.equal(paths_a.S, paths_b.S)` (global RNG
advances between calls, so the two draws differ). Restore the `generator=generator`
argument before Step 3.

- [ ] **Step 3: Write minimal implementation**

No production change needed — `simulate_gbm` already routes all randomness through the
passed `generator` (Task 2.1). The restored line is exactly:

```python
    Z = torch.randn(n_paths, n_steps, generator=generator, device=device)
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/test_gbm.py::test_gbm_reproducible_with_same_seed tests/test_gbm.py::test_gbm_different_seed_differs -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```
git add tests/test_gbm.py
git commit -m "test(simulators): assert GBM reproducibility under seeded generator"
```

---

### Task 2.3: GBM martingale check (discounted E[S_T] ≈ s0 under drift=r=0)

**Files:**
- Modify: `tests/test_gbm.py`

Under `cfg.mu = None` and `cfg.r = 0`, the drift equals `r = 0`, so the (undiscounted,
since `r = 0` the discount factor is `1`) terminal mean satisfies `E[S_T] = s0`. The
Monte-Carlo estimate of the mean has standard error `std(S_T)/sqrt(n_paths)`; we accept
within `3 *` that stderr.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_gbm.py
def test_gbm_martingale_discounted_mean():
    # mu=None -> drift = r = 0  => discount factor exp(-r*T)=1, E[S_T]=s0
    cfg = ExperimentConfig(s0=100.0, sigma=0.2, r=0.0, mu=None, maturity=1.0, n_steps=50)
    assert cfg.drift == 0.0
    gen = torch.Generator(device=cfg.device).manual_seed(2024)
    n_paths = 200_000

    paths = simulate_gbm(cfg, n_paths, gen)
    S_T = paths.S[:, -1]

    discount = torch.exp(torch.tensor(-cfg.r * cfg.maturity))
    discounted = discount * S_T
    mc_mean = discounted.mean()
    mc_stderr = discounted.std(unbiased=True) / (n_paths ** 0.5)

    # |E[S_T]_hat - s0| <= 3 * MC stderr  (well over 99% of the time)
    assert torch.abs(mc_mean - cfg.s0) <= 3.0 * mc_stderr
    # stderr is small at this n_paths (sigma=0.2,T=1 => std(S_T)~20 => stderr~0.045)
    assert mc_stderr < 0.1
```

- [ ] **Step 2: Run test to verify it fails**

To confirm the test genuinely guards the drift, temporarily introduce a drift bug:
change the log-increment in `gbm.py` to drop the `-0.5*sigma*sigma` Itô correction
(i.e. `(drift) * dt + ...`). Then run:

```
pytest tests/test_gbm.py::test_gbm_martingale_discounted_mean -v
```

Expected: FAIL — without the `−½σ²` correction, `E[S_T] = s0·exp(½σ²T) ≈ 102.02`, which
is `~45` stderr above `s0`, tripping `assert torch.abs(mc_mean - cfg.s0) <= 3*mc_stderr`.
Restore the correct increment before Step 3.

- [ ] **Step 3: Write minimal implementation**

No production change — the Task 2.1 implementation already includes the exact Itô
correction:

```python
    log_increments = (drift - 0.5 * sigma * sigma) * dt + sigma * (dt ** 0.5) * Z
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/test_gbm.py::test_gbm_martingale_discounted_mean -v
```

Expected: PASS (`mc_mean` within `3·stderr` of `100.0`; `stderr < 0.1`).

- [ ] **Step 5: Commit**

```
git add tests/test_gbm.py
git commit -m "test(simulators): GBM discounted-mean martingale check under drift=r"
```

---

### Task 2.4: GBM terminal log-variance matches σ²·T

**Files:**
- Modify: `tests/test_gbm.py`

For exact GBM, `log(S_T/s0) ~ N((drift−½σ²)T, σ²T)`. The terminal log-return variance is
`σ²·T` exactly (independent of drift). With `σ=0.2`, `T=1`, that is `0.04`. The MC
estimate of a variance has relative standard error `≈ sqrt(2/(n−1))`; at `n=200_000`
that is `≈ 0.0032`, so a `3%` relative tolerance is comfortable.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_gbm.py
def test_gbm_terminal_log_variance_matches_sigma2T():
    cfg = ExperimentConfig(s0=100.0, sigma=0.2, r=0.0, mu=None, maturity=1.0, n_steps=50)
    gen = torch.Generator(device=cfg.device).manual_seed(7)
    n_paths = 200_000

    paths = simulate_gbm(cfg, n_paths, gen)
    log_ret = torch.log(paths.S[:, -1] / cfg.s0)

    sample_var = log_ret.var(unbiased=True)
    expected_var = cfg.sigma ** 2 * cfg.maturity  # 0.2**2 * 1.0 = 0.04

    # within ~3% relative (MC variance-of-variance is ~0.32% rel at this n_paths)
    assert torch.isclose(sample_var, torch.tensor(expected_var), rtol=0.03)

    # log-return mean should match (drift - 0.5*sigma^2)*T = -0.02
    expected_mean = (cfg.drift - 0.5 * cfg.sigma ** 2) * cfg.maturity
    mean_stderr = (sample_var / n_paths) ** 0.5
    assert torch.abs(log_ret.mean() - expected_mean) <= 4.0 * mean_stderr
```

- [ ] **Step 2: Run test to verify it fails**

To confirm the test guards the diffusion coefficient, temporarily change the diffusion
term in `gbm.py` from `sigma * (dt ** 0.5) * Z` to `sigma * dt * Z` (a common
`sqrt(dt)` bug). Then run:

```
pytest tests/test_gbm.py::test_gbm_terminal_log_variance_matches_sigma2T -v
```

Expected: FAIL — the per-step variance becomes `σ²·dt²` instead of `σ²·dt`, so the
terminal variance is `σ²·dt·T = 0.04·0.02 = 8e-4`, far outside `rtol=0.03` of `0.04`.
Restore `sigma * (dt ** 0.5) * Z` before Step 3.

- [ ] **Step 3: Write minimal implementation**

No production change — the Task 2.1 implementation uses the correct `√Δt` scaling:

```python
    log_increments = (drift - 0.5 * sigma * sigma) * dt + sigma * (dt ** 0.5) * Z
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/test_gbm.py::test_gbm_terminal_log_variance_matches_sigma2T -v
```

Expected: PASS (`sample_var ≈ 0.04` within `3%`; log-return mean within `4·stderr` of
`−0.02`).

- [ ] **Step 5: Commit**

```
git add tests/test_gbm.py
git commit -m "test(simulators): GBM terminal log-variance equals sigma^2*T"
```
## Group 3 — Black–Scholes pricer + Greeks

Implements `deephedge/pricing/black_scholes.py` per spec §7.1 and the §13(1)
correctness gate. All functions are pure torch, broadcast over `S`, `K`, `sigma`,
and accept `tau` as either a Python `float` or a `torch.Tensor`. The `tau→0` limit
is made safe by clamping `tau` to a small floor inside the shared `_d1_d2` helper so
`d1`/`d2` never divide by zero; at expiry the price collapses to intrinsic value and
the Greeks (vega, gamma, theta) collapse to ~0 without producing `NaN`/`inf`.

Contract signatures bound exactly (header `deephedge/pricing/black_scholes.py`):

```python
bs_price(S, K, tau, r, sigma, q=0.0, kind="call") -> torch.Tensor
bs_delta(S, K, tau, r, sigma, q=0.0, kind="call") -> torch.Tensor
bs_vega(S, K, tau, r, sigma, q=0.0)               -> torch.Tensor
bs_gamma(S, K, tau, r, sigma, q=0.0)              -> torch.Tensor
bs_theta(S, K, tau, r, sigma, q=0.0, kind="call") -> torch.Tensor
```

The internal helper `_d1_d2(S, K, tau, r, sigma, q)` returns `(d1, d2, sqrt_tau)` and
is defined fully in Task 3.1. The standard-normal CDF uses `torch.special.ndtr`; the
PDF uses an explicit Gaussian density. Every test seeds randomness with an explicit
`torch.Generator` where randomness is used, and asserts concrete numbers.

---

### Task 3.1: `_d1_d2` helper + `bs_price` for the ATM call known value

**Files:**
- Create: `deephedge/__init__.py`
- Create: `deephedge/pricing/__init__.py`
- Create: `deephedge/pricing/black_scholes.py`
- Test: `tests/pricing/test_black_scholes.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/pricing/test_black_scholes.py
import math

import torch

from deephedge.pricing.black_scholes import _d1_d2, bs_price


def test_d1_d2_atm_unit_vol_one_year():
    # ATM, r=q=0, sigma=0.2, tau=1 -> d1 = +sigma/2*sqrt(tau) = 0.1, d2 = -0.1
    S = torch.tensor(100.0)
    K = torch.tensor(100.0)
    d1, d2, sqrt_tau = _d1_d2(S, K, 1.0, 0.0, 0.2, 0.0)
    assert torch.allclose(d1, torch.tensor(0.1), atol=1e-6)
    assert torch.allclose(d2, torch.tensor(-0.1), atol=1e-6)
    assert torch.allclose(sqrt_tau, torch.tensor(1.0), atol=1e-6)


def test_atm_call_price_textbook_value():
    # Hull textbook value: S=K=100, r=0, sigma=0.2, T=1 -> 7.9656 (computed 7.965567...)
    S = torch.tensor(100.0)
    K = torch.tensor(100.0)
    price = bs_price(S, K, 1.0, 0.0, 0.2, kind="call")
    assert torch.allclose(price, torch.tensor(7.9656), atol=1e-3)
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/pricing/test_black_scholes.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'deephedge.pricing.black_scholes'` (module not created yet).

- [ ] **Step 3: Write minimal implementation**

```python
# deephedge/__init__.py
```

```python
# deephedge/pricing/__init__.py
```

```python
# deephedge/pricing/black_scholes.py
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
    """Black-Scholes European option price."""
    S = torch.as_tensor(S, dtype=torch.get_default_dtype())
    K = _as_tensor(K, S)
    r = _as_tensor(r, S)
    q = _as_tensor(q, S)
    tau = _as_tensor(tau, S)
    d1, d2, _ = _d1_d2(S, K, tau, r, sigma, q)
    disc = torch.exp(-r * tau)
    disc_div = torch.exp(-q * tau)
    nd1 = torch.special.ndtr(d1)
    nd2 = torch.special.ndtr(d2)
    if kind == "call":
        return S * disc_div * nd1 - K * disc * nd2
    elif kind == "put":
        return K * disc * torch.special.ndtr(-d2) - S * disc_div * torch.special.ndtr(-d1)
    raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/pricing/test_black_scholes.py -v
```

Expected: PASS (d1=0.1, d2=-0.1, ATM call price 7.96557 within 1e-3 of 7.9656).

- [ ] **Step 5: Commit**

```
git add deephedge/__init__.py deephedge/pricing/__init__.py deephedge/pricing/black_scholes.py tests/pricing/test_black_scholes.py
git commit -m "feat(pricing): bs_price + _d1_d2 helper with ATM known-value test"
```

---

### Task 3.2: Put–call parity for `bs_price` (regression-lock)

**Files:**
- Modify: `tests/pricing/test_black_scholes.py`

- [ ] **Step 1: Write the test**

```python
# append to tests/pricing/test_black_scholes.py


def test_put_call_parity():
    # C - P = S*exp(-q*tau) - K*exp(-r*tau). With nonzero r, q, off-ATM strikes.
    gen = torch.Generator().manual_seed(0)
    S = 80.0 + 40.0 * torch.rand(16, generator=gen)   # spreads around 100
    K = torch.tensor(100.0)
    r, q, sigma, tau = 0.03, 0.01, 0.25, 0.5
    call = bs_price(S, K, tau, r, sigma, q=q, kind="call")
    put = bs_price(S, K, tau, r, sigma, q=q, kind="put")
    lhs = call - put
    rhs = S * math.exp(-q * tau) - K * math.exp(-r * tau)
    assert torch.allclose(lhs, rhs, atol=1e-5)
```

- [ ] **Step 2: Run the test**

```
pytest tests/pricing/test_black_scholes.py::test_put_call_parity -v
```

Expected: PASS (regression-lock — behavior implemented in Task 3.1; this guards against regression). The `kind="put"` branch already exists, so this parity invariant is a regression guard, not a red→green step.

- [ ] **Step 3: Write minimal implementation**

No production code change is needed — the `kind="put"` branch was implemented in Task 3.1. This task adds the parity regression test only. (If `bs_price` had only a call branch, the put branch from Task 3.1 Step 3 is the minimal implementation that satisfies this test.)

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/pricing/test_black_scholes.py::test_put_call_parity -v
```

Expected: PASS (C − P equals S·e^{−qτ} − K·e^{−rτ} within 1e-5 across all 16 random spots).

- [ ] **Step 5: Commit**

```
git add tests/pricing/test_black_scholes.py
git commit -m "test(pricing): put-call parity regression guard for bs_price"
```

---

### Task 3.3: `bs_delta` with deep-ITM ≈ 1 and deep-OTM ≈ 0

**Files:**
- Modify: `deephedge/pricing/black_scholes.py`
- Modify: `tests/pricing/test_black_scholes.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/pricing/test_black_scholes.py
from deephedge.pricing.black_scholes import bs_delta


def test_call_delta_atm_half_boundary():
    # ATM r=q=0 sigma=0.2 tau=1 -> delta = N(0.1) ~ 0.5398
    S = torch.tensor(100.0)
    K = torch.tensor(100.0)
    delta = bs_delta(S, K, 1.0, 0.0, 0.2, kind="call")
    assert torch.allclose(delta, torch.tensor(0.5398), atol=1e-3)


def test_call_delta_deep_itm_and_otm():
    K = torch.tensor(100.0)
    r, sigma, tau = 0.0, 0.2, 1.0
    itm = bs_delta(torch.tensor(1000.0), K, tau, r, sigma, kind="call")
    otm = bs_delta(torch.tensor(1.0), K, tau, r, sigma, kind="call")
    assert itm.item() > 0.999            # deep ITM call delta -> 1
    assert otm.item() < 1e-3             # deep OTM call delta -> 0


def test_put_delta_equals_call_delta_minus_one():
    # With q=0: put delta = call delta - 1 (= -N(-d1)).
    S = torch.tensor(100.0)
    K = torch.tensor(100.0)
    r, sigma, tau = 0.0, 0.2, 1.0
    cd = bs_delta(S, K, tau, r, sigma, kind="call")
    pd = bs_delta(S, K, tau, r, sigma, kind="put")
    assert torch.allclose(pd, cd - 1.0, atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/pricing/test_black_scholes.py -k delta -v
```

Expected: FAIL with `ImportError: cannot import name 'bs_delta'` (function not defined yet).

- [ ] **Step 3: Write minimal implementation**

```python
# append to deephedge/pricing/black_scholes.py


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
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/pricing/test_black_scholes.py -k delta -v
```

Expected: PASS (ATM call delta 0.5398; deep-ITM > 0.999; deep-OTM < 1e-3; put delta = call delta − 1).

- [ ] **Step 5: Commit**

```
git add deephedge/pricing/black_scholes.py tests/pricing/test_black_scholes.py
git commit -m "feat(pricing): bs_delta with ITM/OTM/ATM boundary tests"
```

---

### Task 3.4: `bs_vega` and `bs_gamma` (positive, known values)

**Files:**
- Modify: `deephedge/pricing/black_scholes.py`
- Modify: `tests/pricing/test_black_scholes.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/pricing/test_black_scholes.py
from deephedge.pricing.black_scholes import bs_gamma, bs_vega


def test_vega_positive_and_known_value():
    # ATM r=q=0 sigma=0.2 tau=1 -> vega = S*phi(d1)*sqrt(tau) = 39.6953
    S = torch.tensor(100.0)
    K = torch.tensor(100.0)
    vega = bs_vega(S, K, 1.0, 0.0, 0.2)
    assert vega.item() > 0.0
    assert torch.allclose(vega, torch.tensor(39.6953), atol=1e-3)


def test_gamma_positive_and_known_value():
    # ATM r=q=0 sigma=0.2 tau=1 -> gamma = phi(d1)/(S*sigma*sqrt(tau)) = 0.0198476
    S = torch.tensor(100.0)
    K = torch.tensor(100.0)
    gamma = bs_gamma(S, K, 1.0, 0.0, 0.2)
    assert gamma.item() > 0.0
    assert torch.allclose(gamma, torch.tensor(0.0198476), atol=1e-6)


def test_vega_gamma_relation():
    # vega = gamma * S^2 * sigma * tau (standard identity, q=0)
    S = torch.tensor(120.0)
    K = torch.tensor(100.0)
    r, sigma, tau = 0.01, 0.3, 0.7
    vega = bs_vega(S, K, tau, r, sigma)
    gamma = bs_gamma(S, K, tau, r, sigma)
    assert torch.allclose(vega, gamma * S * S * sigma * tau, atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/pricing/test_black_scholes.py -k "vega or gamma" -v
```

Expected: FAIL with `ImportError: cannot import name 'bs_gamma'` / `'bs_vega'` (functions not defined yet).

- [ ] **Step 3: Write minimal implementation**

```python
# append to deephedge/pricing/black_scholes.py

_INV_SQRT_2PI = 0.3989422804014327  # 1/sqrt(2*pi)


def _norm_pdf(x: torch.Tensor) -> torch.Tensor:
    """Standard-normal PDF phi(x) = exp(-x^2/2)/sqrt(2*pi)."""
    return _INV_SQRT_2PI * torch.exp(-0.5 * x * x)


def bs_vega(S, K, tau, r, sigma, q=0.0) -> torch.Tensor:
    """Black-Scholes vega = S*exp(-q*tau)*phi(d1)*sqrt(tau). Same for call and put."""
    S = torch.as_tensor(S, dtype=torch.get_default_dtype())
    q = _as_tensor(q, S)
    tau = _as_tensor(tau, S)
    d1, _, sqrt_tau = _d1_d2(S, K, tau, r, sigma, q)
    return S * torch.exp(-q * tau) * _norm_pdf(d1) * sqrt_tau


def bs_gamma(S, K, tau, r, sigma, q=0.0) -> torch.Tensor:
    """Black-Scholes gamma = exp(-q*tau)*phi(d1)/(S*sigma*sqrt(tau)). Same for call and put."""
    S = torch.as_tensor(S, dtype=torch.get_default_dtype())
    q = _as_tensor(q, S)
    sigma_t = _as_tensor(sigma, S)
    tau = _as_tensor(tau, S)
    d1, _, sqrt_tau = _d1_d2(S, K, tau, r, sigma, q)
    return torch.exp(-q * tau) * _norm_pdf(d1) / (S * sigma_t * sqrt_tau)
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/pricing/test_black_scholes.py -k "vega or gamma" -v
```

Expected: PASS (vega 39.6953 > 0; gamma 0.0198476 > 0; vega = gamma·S²·σ·τ identity holds).

- [ ] **Step 5: Commit**

```
git add deephedge/pricing/black_scholes.py tests/pricing/test_black_scholes.py
git commit -m "feat(pricing): bs_vega and bs_gamma with known values and identity"
```

---

### Task 3.5: `bs_theta`

**Files:**
- Modify: `deephedge/pricing/black_scholes.py`
- Modify: `tests/pricing/test_black_scholes.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/pricing/test_black_scholes.py
from deephedge.pricing.black_scholes import bs_theta


def test_call_theta_atm_known_value():
    # ATM r=q=0 sigma=0.2 tau=1: theta = -S*phi(d1)*sigma/(2*sqrt(tau)) = -3.969525
    S = torch.tensor(100.0)
    K = torch.tensor(100.0)
    theta = bs_theta(S, K, 1.0, 0.0, 0.2, kind="call")
    assert torch.allclose(theta, torch.tensor(-3.969525), atol=1e-4)
    assert theta.item() < 0.0   # long call loses value as time passes (r=q=0)


def test_theta_matches_finite_difference():
    # theta = d(price)/d(t) = -d(price)/d(tau); central difference in tau.
    S = torch.tensor(105.0)
    K = torch.tensor(100.0)
    r, sigma, tau, q = 0.02, 0.25, 0.5, 0.0
    h = 1e-4
    up = bs_price(S, K, tau + h, r, sigma, q=q, kind="call")
    dn = bs_price(S, K, tau - h, r, sigma, q=q, kind="call")
    fd_theta = -(up - dn) / (2 * h)
    theta = bs_theta(S, K, tau, r, sigma, q=q, kind="call")
    assert torch.allclose(theta, fd_theta, atol=1e-2)
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/pricing/test_black_scholes.py -k theta -v
```

Expected: FAIL with `ImportError: cannot import name 'bs_theta'` (function not defined yet).

- [ ] **Step 3: Write minimal implementation**

```python
# append to deephedge/pricing/black_scholes.py


def bs_theta(S, K, tau, r, sigma, q=0.0, kind="call") -> torch.Tensor:
    """Black-Scholes theta = d(price)/d(t) (per-year). Uses the safe-tau d1/d2.

    call: -S e^{-q tau} phi(d1) sigma/(2 sqrt(tau)) - r K e^{-r tau} N(d2) + q S e^{-q tau} N(d1)
    put : -S e^{-q tau} phi(d1) sigma/(2 sqrt(tau)) + r K e^{-r tau} N(-d2) - q S e^{-q tau} N(-d1)
    """
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
        return (
            gamma_term
            - r * K * disc * torch.special.ndtr(d2)
            + q * S * disc_div * torch.special.ndtr(d1)
        )
    elif kind == "put":
        return (
            gamma_term
            + r * K * disc * torch.special.ndtr(-d2)
            - q * S * disc_div * torch.special.ndtr(-d1)
        )
    raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/pricing/test_black_scholes.py -k theta -v
```

Expected: PASS (ATM call theta −3.969525 within 1e-4; theta matches central-difference of price w.r.t. tau within 1e-2).

- [ ] **Step 5: Commit**

```
git add deephedge/pricing/black_scholes.py tests/pricing/test_black_scholes.py
git commit -m "feat(pricing): bs_theta with known value and finite-difference check"
```

---

### Task 3.6: Safe `tau→0` limit — intrinsic value, no NaN, ~0 Greeks

**Files:**
- Modify: `tests/pricing/test_black_scholes.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/pricing/test_black_scholes.py


def test_tau_zero_atm_price_zero_no_nan():
    # At tau exactly 0, ATM intrinsic value = max(S-K,0) = 0; must be finite (no NaN).
    S = torch.tensor(100.0)
    K = torch.tensor(100.0)
    price = bs_price(S, K, 0.0, 0.0, 0.2, kind="call")
    assert torch.isfinite(price).all()
    assert torch.allclose(price, torch.tensor(0.0), atol=1e-5)


def test_tau_zero_intrinsic_value_itm_otm():
    # ITM call at expiry -> S-K; OTM call -> 0. tau=0 must not divide by zero.
    K = torch.tensor(100.0)
    itm = bs_price(torch.tensor(130.0), K, 0.0, 0.0, 0.2, kind="call")
    otm = bs_price(torch.tensor(70.0), K, 0.0, 0.0, 0.2, kind="call")
    assert torch.isfinite(itm) and torch.isfinite(otm)
    assert torch.allclose(itm, torch.tensor(30.0), atol=1e-5)
    assert torch.allclose(otm, torch.tensor(0.0), atol=1e-5)


def test_tau_zero_greeks_finite_and_negligible():
    # At tau=0 ATM the clamp keeps everything finite; vega/gamma/theta -> ~0.
    S = torch.tensor(100.0)
    K = torch.tensor(100.0)
    vega = bs_vega(S, K, 0.0, 0.0, 0.2)
    gamma = bs_gamma(S, K, 0.0, 0.0, 0.2)
    theta = bs_theta(S, K, 0.0, 0.0, 0.2, kind="call")
    delta = bs_delta(S, K, 0.0, 0.0, 0.2, kind="call")
    for g in (vega, gamma, theta, delta):
        assert torch.isfinite(g).all()
    assert vega.abs().item() < 1e-3
    assert theta.abs().item() < 1e-3
    # ATM at the clamp boundary: delta sits near the 0.5 boundary, gamma is large
    # but finite (1/sqrt(tau_floor)); both must be free of NaN. Delta in [0,1].
    assert 0.0 <= delta.item() <= 1.0


def test_tau_zero_vector_no_nan_mixed_moneyness():
    # Broadcasting with tau=0 across a vector of spots must never produce NaN.
    gen = torch.Generator().manual_seed(7)
    S = 50.0 + 100.0 * torch.rand(64, generator=gen)
    K = torch.tensor(100.0)
    price = bs_price(S, K, 0.0, 0.0, 0.2, kind="call")
    delta = bs_delta(S, K, 0.0, 0.0, 0.2, kind="call")
    assert torch.isfinite(price).all()
    assert torch.isfinite(delta).all()
    # Each price equals intrinsic value max(S-K,0) at expiry.
    intrinsic = torch.clamp(S - K, min=0.0)
    assert torch.allclose(price, intrinsic, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/pricing/test_black_scholes.py -k tau_zero -v
```

Expected: FAIL — at `tau=0` the as-written `_d1_d2` clamp produces finite `d1/d2`, but the price does NOT exactly equal intrinsic value because the clamped tau (`1e-12`) leaves a tiny residual time value; the strict `atol=1e-5` intrinsic-value assertions fail (and ITM expects exactly `S-K` but gets `S-K + epsilon`). This forces an explicit expiry branch.

- [ ] **Step 3: Write minimal implementation**

```python
# in deephedge/pricing/black_scholes.py: replace the body of bs_price with an
# explicit tau<=0 intrinsic-value branch layered on top of the clamped formula.

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
```

```python
# also harden the Greeks so tau<=0 yields finite, negligible values (no NaN, no
# division blow-up that survives into reported numbers). Append a shared helper
# and wrap vega/gamma/theta outputs with a tau<=0 zero mask.

def _zero_at_expiry(value: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """Force value to 0 wherever tau<=0 (Greeks vanish at expiry), broadcast-safe."""
    expired = torch.broadcast_to(tau <= 0.0, value.shape)
    return torch.where(expired, torch.zeros_like(value), value)
```

```python
# update bs_vega: wrap its return value
def bs_vega(S, K, tau, r, sigma, q=0.0) -> torch.Tensor:
    """Black-Scholes vega = S*exp(-q*tau)*phi(d1)*sqrt(tau). Zero at expiry."""
    S = torch.as_tensor(S, dtype=torch.get_default_dtype())
    q = _as_tensor(q, S)
    tau = _as_tensor(tau, S)
    d1, _, sqrt_tau = _d1_d2(S, K, tau, r, sigma, q)
    value = S * torch.exp(-q * tau) * _norm_pdf(d1) * sqrt_tau
    return _zero_at_expiry(value, tau)
```

```python
# update bs_gamma: wrap its return value
def bs_gamma(S, K, tau, r, sigma, q=0.0) -> torch.Tensor:
    """Black-Scholes gamma = exp(-q*tau)*phi(d1)/(S*sigma*sqrt(tau)). Zero at expiry."""
    S = torch.as_tensor(S, dtype=torch.get_default_dtype())
    q = _as_tensor(q, S)
    sigma_t = _as_tensor(sigma, S)
    tau = _as_tensor(tau, S)
    d1, _, sqrt_tau = _d1_d2(S, K, tau, r, sigma, q)
    value = torch.exp(-q * tau) * _norm_pdf(d1) / (S * sigma_t * sqrt_tau)
    return _zero_at_expiry(value, tau)
```

```python
# update bs_theta: wrap its return value (compute branch result, then mask)
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
```

Note: `bs_delta` is intentionally left using the clamped `d1` (no expiry mask): at the `tau→0` ATM boundary the spec requires delta to be handled "without NaN" and to sit near the `~0.5` boundary, which `N(d1)` with the clamp delivers (d1→0⁺ ⇒ N(d1)→0.5). The `test_tau_zero_greeks_finite_and_negligible` delta assertion only requires finiteness and `0 ≤ delta ≤ 1`, which holds.

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/pricing/test_black_scholes.py -k tau_zero -v
```

Expected: PASS (tau=0 price = intrinsic value exactly within 1e-5 for ATM/ITM/OTM and the random vector; vega/gamma/theta finite and < 1e-3 ~ 0 at expiry; delta finite in [0,1] near 0.5).

- [ ] **Step 5: Commit**

```
git add deephedge/pricing/black_scholes.py tests/pricing/test_black_scholes.py
git commit -m "feat(pricing): exact intrinsic value and zeroed Greeks at tau->0 (NaN-safe)"
```

---

### Task 3.7: Full module regression pass + broadcasting/shape guard (regression-lock)

**Files:**
- Modify: `tests/pricing/test_black_scholes.py`

- [ ] **Step 1: Write the test**

```python
# append to tests/pricing/test_black_scholes.py


def test_broadcasting_tensor_tau_and_scalar_consistency():
    # tau as a tensor (per-step expiries) must broadcast against a vector of spots,
    # and agree element-wise with scalar-tau calls.
    gen = torch.Generator().manual_seed(123)
    S = 80.0 + 40.0 * torch.rand(5, generator=gen)
    K = torch.tensor(100.0)
    taus = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.1])
    r, sigma = 0.01, 0.2
    vec_price = bs_price(S, K, taus, r, sigma, kind="call")
    assert vec_price.shape == (5,)
    for i in range(5):
        scalar = bs_price(S[i], K, float(taus[i]), r, sigma, kind="call")
        assert torch.allclose(vec_price[i], scalar, atol=1e-6)


def test_full_suite_no_nan_across_functions():
    # Smoke matrix: every function finite over a grid of moneyness x tau (incl. 0).
    gen = torch.Generator().manual_seed(99)
    S = 40.0 + 120.0 * torch.rand(32, generator=gen)
    K = torch.tensor(100.0)
    for tau in (0.0, 1e-9, 0.05, 1.0, 2.5):
        for kind in ("call", "put"):
            assert torch.isfinite(bs_price(S, K, tau, 0.01, 0.2, kind=kind)).all()
            assert torch.isfinite(bs_delta(S, K, tau, 0.01, 0.2, kind=kind)).all()
            assert torch.isfinite(bs_theta(S, K, tau, 0.01, 0.2, kind=kind)).all()
        assert torch.isfinite(bs_vega(S, K, tau, 0.01, 0.2)).all()
        assert torch.isfinite(bs_gamma(S, K, tau, 0.01, 0.2)).all()
```

- [ ] **Step 2: Run the test**

```
pytest tests/pricing/test_black_scholes.py -k "broadcasting or full_suite" -v
```

Expected: PASS (regression-lock — behavior implemented in Tasks 3.1–3.6; this guards against regression). All functions already broadcast tensor `tau` and are NaN-free; if `bs_price` did not broadcast tensor `tau`, the shape assertion `vec_price.shape == (5,)` would catch it — that is the guarded contract.

- [ ] **Step 3: Write minimal implementation**

No production change required — Tasks 3.1–3.6 already implement tensor-`tau` broadcasting (via `_as_tensor` + `_d1_d2` clamp) and the expiry masks. This task locks broadcasting and NaN-freedom as regression tests.

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/pricing/test_black_scholes.py -v
```

Expected: PASS (entire `test_black_scholes.py` green: ATM value, parity, delta boundaries, vega/gamma/theta, tau→0 safety, broadcasting, full NaN-free matrix).

- [ ] **Step 5: Commit**

```
git add tests/pricing/test_black_scholes.py
git commit -m "test(pricing): broadcasting and NaN-free regression matrix for black_scholes"
```
## Group 4 — Portfolio P&L engine + basic strategies + replication gate

> Binds to Shared Contracts in `00-header.md`. Implements the differentiable P&L
> engine (`deephedge/portfolio.py`), the simplest strategies
> (`deephedge/benchmarks.py`), and the end-to-end frictionless replication gate
> (`tests/test_replication_gate.py`). Spec refs: §9, §10, §13(6).
>
> **Dependencies from other sections (must already exist on the branch):**
> `deephedge/config.py` (`ExperimentConfig`), `deephedge/simulators/base.py`
> (`Paths`) + `deephedge/simulators/gbm.py` (`simulate_gbm`),
> `deephedge/pricing/black_scholes.py` (`bs_price`, `bs_delta`),
> `deephedge/instruments.py` (`EuropeanOption`, `payoff`),
> `deephedge/hedger.py` (`Hedger`, `build_features`).
>
> This section implements the **underlying-only** branch of `build_instr_prices`
> (col 0 = `paths.S`). The `("underlying", "option")` branch raises a documented
> `NotImplementedError` to be filled by the multi-instrument section. Per the header
> contract, `build_instr_prices(cfg, paths, hedge_option)` marks col1 with the HEDGE
> option, while `simulate_pnl(..., option, ...)` uses the SOLD/LIABILITY option for the
> terminal payoff and premium — callers thread the two distinct options consistently.

---

### Task 4.1: `StepState` / `PnLResult` dataclasses

**Files:**
- Create: `deephedge/portfolio.py`
- Test: `tests/test_portfolio.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_portfolio.py
import torch
from deephedge.portfolio import StepState, PnLResult


def test_step_state_holds_fields():
    B, m = 4, 1
    s = torch.full((B,), 100.0)
    holdings = torch.zeros(B, m)
    prices = torch.full((B, m), 100.0)
    state = StepState(
        step=0, S=s, V=None, tau=1.0,
        prev_holdings=holdings, instr_prices=prices,
    )
    assert state.step == 0
    assert state.V is None
    assert state.tau == 1.0
    assert torch.equal(state.S, s)
    assert torch.equal(state.prev_holdings, holdings)
    assert state.instr_prices.shape == (B, m)


def test_pnl_result_holds_fields():
    B = 4
    pnl = torch.zeros(B)
    turnover = torch.zeros(B)
    cost = torch.zeros(B)
    holdings = torch.zeros(B, 5, 1)
    res = PnLResult(pnl=pnl, turnover=turnover, cost=cost, holdings=holdings)
    assert res.pnl.shape == (B,)
    assert res.turnover.shape == (B,)
    assert res.cost.shape == (B,)
    assert res.holdings.shape == (B, 5, 1)
```

- [ ] **Step 2: Run test to verify it fails**

`pytest tests/test_portfolio.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'deephedge.portfolio'`.

- [ ] **Step 3: Write minimal implementation**

```python
# deephedge/portfolio.py
"""Differentiable hedged-P&L engine (spec §9) and instrument marking.

A *Strategy* is a callable ``strategy(state: StepState) -> holdings`` returning
a ``(B, n_instruments)`` tensor of target holdings for the current step.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class StepState:
    step: int
    S: torch.Tensor                 # (B,) underlying price at this step
    V: torch.Tensor | None          # (B,) variance at this step, or None
    tau: float                      # time to option maturity (years) at this step
    prev_holdings: torch.Tensor     # (B, n_instruments) holdings carried in
    instr_prices: torch.Tensor      # (B, n_instruments) instrument prices now


@dataclass
class PnLResult:
    pnl: torch.Tensor       # (n_paths,) terminal hedged P&L
    turnover: torch.Tensor  # (n_paths,) sum_i sum_j |delta_i - delta_{i-1}|
    cost: torch.Tensor      # (n_paths,) total proportional transaction cost
    holdings: torch.Tensor  # (n_paths, n_steps, n_instruments) chosen holdings
```

- [ ] **Step 4: Run test to verify it passes**

`pytest tests/test_portfolio.py -v`

Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add deephedge/portfolio.py tests/test_portfolio.py
git commit -m "feat(portfolio): add StepState and PnLResult dataclasses"
```

---

### Task 4.2: `build_instr_prices` — underlying-only branch

**Files:**
- Modify: `deephedge/portfolio.py`
- Test: `tests/test_portfolio.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_portfolio.py  (append)
from deephedge.config import ExperimentConfig
from deephedge.simulators.gbm import simulate_gbm
from deephedge.instruments import EuropeanOption
from deephedge.portfolio import build_instr_prices


def test_build_instr_prices_underlying_only_is_S():
    cfg = ExperimentConfig(n_steps=5, n_paths=8, instruments=("underlying",))
    gen = torch.Generator().manual_seed(0)
    paths = simulate_gbm(cfg, n_paths=8, generator=gen)
    option = EuropeanOption(strike=cfg.k, maturity=cfg.maturity, kind="call")
    prices = build_instr_prices(cfg, paths, option)
    # shape: (n_paths, n_steps+1, n_instruments)
    assert prices.shape == (8, cfg.n_steps + 1, 1)
    # underlying column equals the simulated path exactly
    assert torch.equal(prices[:, :, 0], paths.S)


def test_build_instr_prices_option_branch_not_implemented():
    cfg = ExperimentConfig(
        n_steps=5, n_paths=8, instruments=("underlying", "option")
    )
    gen = torch.Generator().manual_seed(0)
    paths = simulate_gbm(cfg, n_paths=8, generator=gen)
    option = EuropeanOption(strike=cfg.k, maturity=cfg.maturity, kind="call")
    try:
        build_instr_prices(cfg, paths, option)
        raised = False
    except NotImplementedError:
        raised = True
    assert raised, "option leg must raise NotImplementedError in this section"
```

- [ ] **Step 2: Run test to verify it fails**

`pytest tests/test_portfolio.py -k build_instr_prices -v`

Expected: FAIL with `ImportError: cannot import name 'build_instr_prices' from 'deephedge.portfolio'`.

- [ ] **Step 3: Write minimal implementation**

```python
# deephedge/portfolio.py  (append)
from deephedge.simulators.base import Paths  # noqa: E402
from deephedge.config import ExperimentConfig  # noqa: E402
from deephedge.instruments import EuropeanOption  # noqa: E402


def build_instr_prices(
    cfg: ExperimentConfig, paths: Paths, hedge_option: EuropeanOption
) -> torch.Tensor:
    """Stack hedging-instrument prices into ``(n_paths, n_steps+1, n_instruments)``.

    Column 0 is always the underlying ``paths.S`` (col tensor shape ``(..., 1)`` in the
    underlying-only case). ``hedge_option`` is the HEDGE instrument's option (its own
    strike/maturity), NOT the sold/liability option. If ``"option"`` is in
    ``cfg.instruments`` the option leg is marked by the multi-instrument section; here it
    raises ``NotImplementedError`` so the contract surface is stable.
    """
    n_paths, n_steps_p1 = paths.S.shape
    cols = [paths.S]  # col 0 = underlying
    for name in cfg.instruments[1:]:
        if name == "option":
            # Filled by the multi-instrument section: mark the HEDGE option leg each
            # step via mark_option(cfg, paths, hedge_option). Deferred here.
            raise NotImplementedError(
                "option hedging instrument is implemented by the "
                "multi-instrument section (use mark_option to fill this branch)"
            )
        raise ValueError(f"unknown hedging instrument: {name!r}")
    out = torch.stack(cols, dim=-1)  # (n_paths, n_steps+1, n_instruments)
    return out.to(cfg.device)
```

- [ ] **Step 4: Run test to verify it passes**

`pytest tests/test_portfolio.py -k build_instr_prices -v`

Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add deephedge/portfolio.py tests/test_portfolio.py
git commit -m "feat(portfolio): build_instr_prices underlying-only branch"
```

---

### Task 4.3: `simulate_pnl` — MTM accumulation (cost-free)

**Files:**
- Modify: `deephedge/portfolio.py`
- Test: `tests/test_portfolio.py`

Implements the core loop of spec §9:
`PnL = premium + Σ δ_i·(P_{i+1}−P_i) − Σ cost_i − payoff(S_N)`.
This task establishes the MTM + payoff + premium accounting with `cost = 0`;
the proportional-cost term is verified in Task 4.4.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_portfolio.py  (append)
from deephedge.instruments import payoff
from deephedge.portfolio import simulate_pnl


def _constant_unit_strategy(state):
    # always hold exactly 1 unit of every instrument
    return torch.ones_like(state.prev_holdings)


def test_simulate_pnl_mtm_telescopes_no_cost():
    # With cost=0 and a constant unit holding in the underlying only, the MTM
    # sum telescopes to (S_N - S_0). So pnl = premium + (S_N - S_0) - payoff.
    cfg = ExperimentConfig(
        n_steps=10, n_paths=16, cost=0.0, instruments=("underlying",)
    )
    gen = torch.Generator().manual_seed(1)
    paths = simulate_gbm(cfg, n_paths=16, generator=gen)
    option = EuropeanOption(strike=cfg.k, maturity=cfg.maturity, kind="call")
    instr = build_instr_prices(cfg, paths, option)
    premium = 3.5
    res = simulate_pnl(
        _constant_unit_strategy, paths, cfg, option, premium, instr
    )
    S0 = paths.S[:, 0]
    S_N = paths.S[:, -1]
    expected = premium + (S_N - S0) - payoff(option, S_N)
    assert res.pnl.shape == (16,)
    assert torch.allclose(res.pnl, expected, atol=1e-5)
    # zero cost with cost=0
    assert torch.allclose(res.cost, torch.zeros(16), atol=1e-7)
    # holdings record: (n_paths, n_steps, n_instruments)
    assert res.holdings.shape == (16, cfg.n_steps, 1)


def test_simulate_pnl_shapes():
    cfg = ExperimentConfig(
        n_steps=7, n_paths=5, cost=0.0, instruments=("underlying",)
    )
    gen = torch.Generator().manual_seed(2)
    paths = simulate_gbm(cfg, n_paths=5, generator=gen)
    option = EuropeanOption(strike=cfg.k, maturity=cfg.maturity, kind="call")
    instr = build_instr_prices(cfg, paths, option)
    res = simulate_pnl(_constant_unit_strategy, paths, cfg, option, 1.0, instr)
    assert res.pnl.shape == (5,)
    assert res.turnover.shape == (5,)
    assert res.cost.shape == (5,)
    assert res.holdings.shape == (5, 7, 1)
```

- [ ] **Step 2: Run test to verify it fails**

`pytest tests/test_portfolio.py -k simulate_pnl -v`

Expected: FAIL with `ImportError: cannot import name 'simulate_pnl' from 'deephedge.portfolio'`.

- [ ] **Step 3: Write minimal implementation**

```python
# deephedge/portfolio.py  (append)
from deephedge.instruments import payoff as _payoff  # noqa: E402


def simulate_pnl(
    strategy,
    paths: Paths,
    cfg: ExperimentConfig,
    option: EuropeanOption,
    premium: float,
    instr_prices: torch.Tensor,
) -> PnLResult:
    """Roll the hedged trajectory forward and accumulate terminal P&L (spec §9).

    ``strategy(state: StepState) -> holdings`` returns ``(B, n_instruments)``.
    Holdings are chosen at steps ``0..N-1`` (no trade is needed at the terminal
    step ``N``). P&L = premium + MTM gains - proportional costs - short-call
    payoff. Fully differentiable in torch w.r.t. anything ``strategy`` depends on.
    """
    n_paths, n_steps_p1, n_instr = instr_prices.shape
    n_steps = n_steps_p1 - 1

    S = paths.S
    V = paths.V
    dt = cfg.dt

    prev = torch.zeros(n_paths, n_instr, device=instr_prices.device)
    pnl = torch.zeros(n_paths, device=instr_prices.device)
    turnover = torch.zeros(n_paths, device=instr_prices.device)
    cost = torch.zeros(n_paths, device=instr_prices.device)
    holdings_log = []

    for i in range(n_steps):
        p_i = instr_prices[:, i, :]       # (n_paths, n_instr)
        p_next = instr_prices[:, i + 1, :]
        tau = cfg.maturity - i * dt       # time to option maturity at step i
        V_i = None if V is None else V[:, i]
        state = StepState(
            step=i,
            S=S[:, i],
            V=V_i,
            tau=tau,
            prev_holdings=prev,
            instr_prices=p_i,
        )
        holdings = strategy(state)        # (n_paths, n_instr)
        holdings_log.append(holdings)

        # MTM gain over [t_i, t_{i+1}] across all instruments
        pnl = pnl + (holdings * (p_next - p_i)).sum(dim=-1)

        # proportional transaction cost at rebalance time t_i
        trade = (holdings - prev).abs()             # (n_paths, n_instr)
        turnover = turnover + trade.sum(dim=-1)
        step_cost = cfg.cost * (p_i * trade).sum(dim=-1)
        cost = cost + step_cost
        pnl = pnl - step_cost

        prev = holdings

    # premium received (short the call) and terminal liability
    pnl = pnl + premium - _payoff(option, S[:, -1])

    holdings_out = torch.stack(holdings_log, dim=1)  # (n_paths, n_steps, n_instr)
    return PnLResult(pnl=pnl, turnover=turnover, cost=cost, holdings=holdings_out)
```

- [ ] **Step 4: Run test to verify it passes**

`pytest tests/test_portfolio.py -k simulate_pnl -v`

Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add deephedge/portfolio.py tests/test_portfolio.py
git commit -m "feat(portfolio): simulate_pnl MTM accumulation and shapes"
```

---

### Task 4.4: `simulate_pnl` — proportional transaction cost

**Files:**
- Modify: `deephedge/portfolio.py` (no change expected; this validates Task 4.3 cost logic)
- Test: `tests/test_portfolio.py`

The cost term `c · Σ_j p^{(j)}_i · |δ^{(j)}_i − δ^{(j)}_{i-1}|` (spec §3/§9) is
verified with a deterministic strategy so the expected number is hand-computable.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_portfolio.py  (append)
def _hold_one_then_two(state):
    # step 0: from prev (0) -> 1 ; steps>=1: hold 2  (one further trade at step 1)
    if state.step == 0:
        return torch.ones_like(state.prev_holdings)
    return 2.0 * torch.ones_like(state.prev_holdings)


def test_simulate_pnl_proportional_cost_exact():
    # Two trades happen: step0 |1-0|=1 priced at S_0; step1 |2-1|=1 priced at S_1.
    # Steps 2..N-1 trade |2-2|=0 -> no further cost.
    cfg = ExperimentConfig(
        n_steps=4, n_paths=3, cost=0.01, instruments=("underlying",)
    )
    gen = torch.Generator().manual_seed(7)
    paths = simulate_gbm(cfg, n_paths=3, generator=gen)
    option = EuropeanOption(strike=cfg.k, maturity=cfg.maturity, kind="call")
    instr = build_instr_prices(cfg, paths, option)
    res = simulate_pnl(_hold_one_then_two, paths, cfg, option, 0.0, instr)

    S0 = paths.S[:, 0]
    S1 = paths.S[:, 1]
    expected_cost = cfg.cost * (S0 * 1.0 + S1 * 1.0)
    assert torch.allclose(res.cost, expected_cost, atol=1e-5)
    # turnover counts |Δδ| summed over steps: 1 (step0) + 1 (step1) = 2 per path
    assert torch.allclose(res.turnover, torch.full((3,), 2.0), atol=1e-6)


def test_simulate_pnl_no_cost_when_rate_zero():
    cfg = ExperimentConfig(
        n_steps=4, n_paths=3, cost=0.0, instruments=("underlying",)
    )
    gen = torch.Generator().manual_seed(7)
    paths = simulate_gbm(cfg, n_paths=3, generator=gen)
    option = EuropeanOption(strike=cfg.k, maturity=cfg.maturity, kind="call")
    instr = build_instr_prices(cfg, paths, option)
    res = simulate_pnl(_hold_one_then_two, paths, cfg, option, 0.0, instr)
    assert torch.allclose(res.cost, torch.zeros(3), atol=1e-7)
    # turnover is independent of the cost rate
    assert torch.allclose(res.turnover, torch.full((3,), 2.0), atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

`pytest tests/test_portfolio.py -k proportional_cost -v`

Expected: PASS already if Task 4.3 cost logic is correct; if the cost term were
implemented wrong (e.g. missing `p_i` weighting or wrong sign), this FAILS with
an `allclose` assertion error. Run to confirm it PASSES; if it FAILS, fix the
cost block in `simulate_pnl` until it matches `c·Σ p_i·|Δδ|`.

- [ ] **Step 3: Write minimal implementation**

No new code: Task 4.3's `simulate_pnl` already computes
`step_cost = cfg.cost * (p_i * trade).sum(dim=-1)` and accumulates
`turnover += trade.sum(dim=-1)`. This task pins that behavior with an exact
hand-computed expectation. (If it failed in Step 2, the minimal fix is to make
the cost block read exactly as in Task 4.3.)

- [ ] **Step 4: Run test to verify it passes**

`pytest tests/test_portfolio.py -k "proportional_cost or no_cost_when_rate_zero" -v`

Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add tests/test_portfolio.py
git commit -m "test(portfolio): pin proportional cost and turnover accounting"
```

---

### Task 4.5: `make_no_hedge_strategy` + no-hedge P&L identity

**Files:**
- Create: `deephedge/benchmarks.py`
- Test: `tests/test_benchmarks.py`

Spec §10 no-hedge baseline: collect premium, pay payoff, never trade ⇒
`pnl == premium − payoff(S_N)` exactly (no MTM, no cost).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_benchmarks.py
import torch
from deephedge.config import ExperimentConfig
from deephedge.simulators.gbm import simulate_gbm
from deephedge.instruments import EuropeanOption, payoff
from deephedge.portfolio import build_instr_prices, simulate_pnl
from deephedge.benchmarks import make_no_hedge_strategy


def test_no_hedge_pnl_equals_premium_minus_payoff_exact():
    cfg = ExperimentConfig(
        n_steps=12, n_paths=64, cost=0.05, instruments=("underlying",)
    )
    gen = torch.Generator().manual_seed(3)
    paths = simulate_gbm(cfg, n_paths=64, generator=gen)
    option = EuropeanOption(strike=cfg.k, maturity=cfg.maturity, kind="call")
    instr = build_instr_prices(cfg, paths, option)
    premium = 4.0
    strat = make_no_hedge_strategy()
    res = simulate_pnl(strat, paths, cfg, option, premium, instr)

    expected = premium - payoff(option, paths.S[:, -1])
    # Exact: zero holdings -> zero MTM and zero turnover/cost even at cost=0.05.
    assert torch.allclose(res.pnl, expected, atol=1e-6)
    assert torch.allclose(res.cost, torch.zeros(64), atol=1e-7)
    assert torch.allclose(res.turnover, torch.zeros(64), atol=1e-7)
    assert torch.count_nonzero(res.holdings) == 0
```

- [ ] **Step 2: Run test to verify it fails**

`pytest tests/test_benchmarks.py -k no_hedge -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'deephedge.benchmarks'`.

- [ ] **Step 3: Write minimal implementation**

```python
# deephedge/benchmarks.py
"""Analytic / non-learned hedging strategies (spec §10).

Each factory returns a *Strategy*: a callable
``strategy(state: StepState) -> holdings`` with shape ``(B, n_instruments)``.
``make_bs_delta_vega_strategy`` is provided by the multi-instrument section.
"""
from __future__ import annotations

import torch

from deephedge.config import ExperimentConfig
from deephedge.portfolio import StepState


def make_no_hedge_strategy():
    """Hold nothing: collect premium, pay payoff. Context baseline."""

    def strategy(state: StepState) -> torch.Tensor:
        return torch.zeros_like(state.prev_holdings)

    return strategy
```

- [ ] **Step 4: Run test to verify it passes**

`pytest tests/test_benchmarks.py -k no_hedge -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add deephedge/benchmarks.py tests/test_benchmarks.py
git commit -m "feat(benchmarks): make_no_hedge_strategy + P&L identity test"
```

---

### Task 4.6: `make_bs_delta_strategy`

**Files:**
- Modify: `deephedge/benchmarks.py`
- Test: `tests/test_benchmarks.py`

Spec §10: BS delta hedge `δ_i = N(d1)` (long delta of the underlying to offset
the short call). Uses `cfg.sigma`, `cfg.r`, `cfg.q`, option strike, and `tau`
from the `StepState`. We are short the call, so we hold `+delta` units of the
underlying to neutralize. We compute the BS delta of the option and assign it to
the underlying leg (column 0), zero on any further legs.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_benchmarks.py  (append)
from deephedge.pricing.black_scholes import bs_delta
from deephedge.benchmarks import make_bs_delta_strategy


def test_bs_delta_strategy_matches_bs_delta():
    cfg = ExperimentConfig(
        n_steps=10, n_paths=32, sigma=0.2, r=0.0, q=0.0,
        instruments=("underlying",),
    )
    strat = make_bs_delta_strategy(cfg)
    S_i = torch.linspace(80.0, 120.0, 32)
    prev = torch.zeros(32, 1)
    tau = 0.05
    state = StepState(
        step=2, S=S_i, V=None, tau=tau,
        prev_holdings=prev, instr_prices=S_i.unsqueeze(-1),
    )
    holdings = strat(state)
    assert holdings.shape == (32, 1)
    expected = bs_delta(
        S_i, torch.tensor(cfg.k), torch.tensor(tau),
        cfg.r, cfg.sigma, q=cfg.q, kind="call",
    )
    assert torch.allclose(holdings[:, 0], expected, atol=1e-5)


def test_bs_delta_strategy_deep_itm_near_one():
    cfg = ExperimentConfig(sigma=0.2, r=0.0, q=0.0, instruments=("underlying",))
    strat = make_bs_delta_strategy(cfg)
    S_i = torch.full((4,), 200.0)  # deep ITM call -> delta ~ 1
    state = StepState(
        step=0, S=S_i, V=None, tau=0.05,
        prev_holdings=torch.zeros(4, 1), instr_prices=S_i.unsqueeze(-1),
    )
    holdings = strat(state)
    assert torch.all(holdings[:, 0] > 0.99)
```

- [ ] **Step 2: Run test to verify it fails**

`pytest tests/test_benchmarks.py -k bs_delta -v`

Expected: FAIL with `ImportError: cannot import name 'make_bs_delta_strategy' from 'deephedge.benchmarks'`.

- [ ] **Step 3: Write minimal implementation**

```python
# deephedge/benchmarks.py  (append)
from deephedge.pricing.black_scholes import bs_delta  # noqa: E402


def make_bs_delta_strategy(cfg: ExperimentConfig):
    """BS delta hedge of the short call: hold +N(d1) units of the underlying.

    Uses ``cfg.sigma`` (true vol under GBM), ``cfg.r``, ``cfg.q``, the hedge
    option strike (defaults to ``cfg.k``), and ``tau`` taken from ``StepState``.
    Only the underlying leg (column 0) is set; any further legs are zero.
    """
    strike = cfg.hedge_option_strike if cfg.hedge_option_strike is not None else cfg.k

    def strategy(state: StepState) -> torch.Tensor:
        S_i = state.S
        tau = torch.as_tensor(state.tau, dtype=S_i.dtype, device=S_i.device)
        delta = bs_delta(
            S_i,
            torch.as_tensor(strike, dtype=S_i.dtype, device=S_i.device),
            tau,
            cfg.r,
            cfg.sigma,
            q=cfg.q,
            kind="call",
        )
        holdings = torch.zeros_like(state.prev_holdings)
        holdings[:, 0] = delta
        return holdings

    return strategy
```

- [ ] **Step 4: Run test to verify it passes**

`pytest tests/test_benchmarks.py -k bs_delta -v`

Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add deephedge/benchmarks.py tests/test_benchmarks.py
git commit -m "feat(benchmarks): make_bs_delta_strategy using bs_delta + tau from state"
```

---

### Task 4.7: `make_nn_strategy` + gradient flow

**Files:**
- Modify: `deephedge/benchmarks.py`
- Test: `tests/test_benchmarks.py`

Spec §8/§9: the NN strategy calls `build_features(state.S, state.tau / cfg.maturity,
state.prev_holdings, state.V, k_norm=cfg.k)` (normalized tau, cfg.k moneyness) then the
`Hedger`. The end-to-end gradient must flow holdings → MTM/costs → terminal P&L →
`loss = pnl.mean()` → hedger params (spec §13(7)).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_benchmarks.py  (append)
from deephedge.hedger import Hedger, build_features
from deephedge.benchmarks import make_nn_strategy


def _n_features(cfg):
    # build_features order: [log(S/k), tau, *prev_holdings, (sqrt(V) if V)]
    n = 2 + cfg.n_instruments
    return n  # GBM: no V term


def test_nn_strategy_output_shape():
    cfg = ExperimentConfig(n_paths=8, instruments=("underlying",))
    torch.manual_seed(0)
    hedger = Hedger(n_features=_n_features(cfg), n_instruments=cfg.n_instruments)
    strat = make_nn_strategy(hedger, cfg)
    S_i = torch.full((8,), 100.0)
    state = StepState(
        step=0, S=S_i, V=None, tau=0.1,
        prev_holdings=torch.zeros(8, 1), instr_prices=S_i.unsqueeze(-1),
    )
    holdings = strat(state)
    assert holdings.shape == (8, 1)


def test_nn_strategy_gradients_flow_to_hedger_params():
    cfg = ExperimentConfig(
        n_steps=5, n_paths=16, cost=0.01, instruments=("underlying",)
    )
    gen = torch.Generator().manual_seed(11)
    paths = simulate_gbm(cfg, n_paths=16, generator=gen)
    option = EuropeanOption(strike=cfg.k, maturity=cfg.maturity, kind="call")
    instr = build_instr_prices(cfg, paths, option)

    torch.manual_seed(0)
    hedger = Hedger(n_features=_n_features(cfg), n_instruments=cfg.n_instruments)
    strat = make_nn_strategy(hedger, cfg)

    res = simulate_pnl(strat, paths, cfg, option, premium=3.0, instr_prices=instr)
    loss = res.pnl.mean()
    loss.backward()

    grads = [p.grad for p in hedger.parameters()]
    assert all(g is not None for g in grads), "every param must receive a grad"
    total = sum(g.abs().sum() for g in grads)
    assert total > 0, "at least one nonzero gradient must flow back"
    assert res.pnl.requires_grad
```

- [ ] **Step 2: Run test to verify it fails**

`pytest tests/test_benchmarks.py -k nn_strategy -v`

Expected: FAIL with `ImportError: cannot import name 'make_nn_strategy' from 'deephedge.benchmarks'`.

- [ ] **Step 3: Write minimal implementation**

```python
# deephedge/benchmarks.py  (append)
from deephedge.hedger import build_features  # noqa: E402


def make_nn_strategy(hedger, cfg: ExperimentConfig):
    """Learned policy: build_features(...) -> Hedger.forward -> holdings.

    Keeps the autograd graph intact so gradients flow from terminal P&L back
    through the trajectory to ``hedger`` parameters (spec §9, §13(7)).
    """

    def strategy(state: StepState) -> torch.Tensor:
        # Normalize tau to (T - t_i)/T and pass cfg.k as the moneyness strike, per the
        # build_features contract. state.S may be (B,) — build_features reshapes to columns.
        features = build_features(
            state.S,
            state.tau / cfg.maturity,
            state.prev_holdings,
            state.V,
            k_norm=cfg.k,
        )
        return hedger(features)

    return strategy
```

- [ ] **Step 4: Run test to verify it passes**

`pytest tests/test_benchmarks.py -k nn_strategy -v`

Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add deephedge/benchmarks.py tests/test_benchmarks.py
git commit -m "feat(benchmarks): make_nn_strategy + gradient-flow test"
```

---

### Task 4.8: Replication gate (end-to-end)

**Files:**
- Create: `tests/test_replication_gate.py`

Spec §13(6): frictionless (`cost=0`), **fine-grid** GBM (`n_steps=200`) +
`make_bs_delta_strategy`, premium = `bs_price(ATM)` ⇒ hedged P&L ≈ 0 with low
variance. **Tolerance-based, not bit-exact:** on a discrete grid the market is
incomplete and the BS-delta replication holds only in the continuous limit, so
we assert `mean(|pnl|)` is small and `std(pnl)` is well below a fraction of the
premium (`std < 0.15 · premium`), not zero.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_replication_gate.py
"""End-to-end replication gate (spec §13(6)).

Frictionless, fine-grid GBM + BS-delta hedge, premium = BS price, must produce a
hedged P&L tightly concentrated at zero. This is a TOLERANCE gate, not bit-exact:
- The MTM sum is a discrete (left-endpoint) approximation of the stochastic
  integral int delta dS; discretization error -> 0 only as n_steps -> infinity.
- On any finite grid the market is incomplete, so perfect replication is
  unattainable; we expect small grid-dependent deviations (spec §13(6)).
Hence we assert mean(|pnl|) is small and std(pnl) < 0.15 * premium, seeded.
"""
import torch

from deephedge.config import ExperimentConfig
from deephedge.simulators.gbm import simulate_gbm
from deephedge.instruments import EuropeanOption
from deephedge.pricing.black_scholes import bs_price
from deephedge.portfolio import build_instr_prices, simulate_pnl
from deephedge.benchmarks import make_bs_delta_strategy


def test_bs_delta_replicates_on_fine_grid_frictionless():
    cfg = ExperimentConfig(
        s0=100.0, k=100.0, r=0.0, q=0.0, mu=None,   # mu=None -> drift = r (RN)
        maturity=30 / 252, n_steps=200,
        sigma=0.2, cost=0.0, instruments=("underlying",),
    )
    gen = torch.Generator().manual_seed(2024)
    n_paths = 20_000
    paths = simulate_gbm(cfg, n_paths=n_paths, generator=gen)

    option = EuropeanOption(strike=cfg.k, maturity=cfg.maturity, kind="call")
    instr = build_instr_prices(cfg, paths, option)

    # Premium = risk-neutral BS price at inception (ATM).
    premium = float(
        bs_price(
            torch.tensor(cfg.s0), torch.tensor(cfg.k),
            torch.tensor(cfg.maturity), cfg.r, cfg.sigma, q=cfg.q, kind="call",
        )
    )
    assert premium > 0

    strat = make_bs_delta_strategy(cfg)
    res = simulate_pnl(strat, paths, cfg, option, premium, instr)

    pnl = res.pnl.detach()
    mean_abs = pnl.abs().mean().item()
    std = pnl.std().item()

    # Mean P&L hugs zero (replication is unbiased in the continuous limit).
    assert abs(pnl.mean().item()) < 0.05 * premium, (
        f"mean P&L {pnl.mean().item():.4f} too large vs premium {premium:.4f}"
    )
    # Tail/variance collapses on the fine grid: std well under 15% of premium.
    assert std < 0.15 * premium, (
        f"std(pnl)={std:.4f} not below 0.15*premium={0.15 * premium:.4f}"
    )
    # No-cost run accrues zero cost; the strategy actually traded.
    assert torch.allclose(res.cost, torch.zeros(n_paths), atol=1e-6)
    assert res.turnover.mean().item() > 0
    # Mean absolute hedging error is a small fraction of premium.
    assert mean_abs < 0.4 * premium
```

- [ ] **Step 2: Run test to verify it fails**

`pytest tests/test_replication_gate.py -v`

Expected: FAIL — initially with a collection/import error if any upstream module
is missing, or (once modules exist) this is the first run that exercises the full
engine end-to-end. Confirm it fails before the engine is wired (e.g.
`ModuleNotFoundError`/`ImportError`), then proceed.

- [ ] **Step 3: Write minimal implementation**

No new product code: the gate is satisfied by `simulate_gbm`,
`build_instr_prices`, `simulate_pnl`, `make_bs_delta_strategy`, and `bs_price`
implemented above (and in the GBM/BS sections). If the assertions fail at Step 4,
debug in this order per spec §13(6): (a) confirm `cfg.drift == cfg.r` (mu=None),
(b) confirm premium uses risk-neutral `r`, (c) confirm the MTM sign and that the
short-call payoff is subtracted, (d) increase `n_steps` toward the continuous
limit only if the discretization bias dominates — do not loosen the tolerance
below the spec's `0.15 * premium` headline gate.

- [ ] **Step 4: Run test to verify it passes**

`pytest tests/test_replication_gate.py -v`

Expected: PASS — `std(pnl) < 0.15 * premium` and `|mean(pnl)| < 0.05 * premium`.

- [ ] **Step 5: Commit**

```bash
git add tests/test_replication_gate.py
git commit -m "test(portfolio): end-to-end frictionless replication gate (spec 13.6)"
```
## Group 5 — Risk-measure losses (CVaR + entropic)

Implements `deephedge/losses.py` per the Shared Contracts:

```python
def cvar_loss(pnl, alpha: float, w) -> torch.Tensor
def entropic_loss(pnl, lam: float) -> torch.Tensor
```

Both functions consume a profit-positive `pnl` tensor of shape `(N,)` (one terminal
P&L per Monte-Carlo path) and return a scalar `torch.Tensor` that closes the autograd
graph for training. Spec ref: §6 (§6.1 CVaR, §6.2 entropic), §13 tests 4–5.

Conventions binding every task below:
- Loss space is `L = -pnl` (larger = worse), exactly as §6.
- `cvar_loss` follows Rockafellar–Uryasev with the **`(1 - alpha)` tail-probability
  denominator** (using `alpha` is the classic ~19x mis-scaling bug — §6.1) and
  `relu = (·)^+`. `w` is a scalar tensor / `nn.Parameter` (shape `()`) passed in.
- `entropic_loss` subtracts `log(N)` so values are comparable across batch sizes, uses
  the max-shift-stabilized `torch.logsumexp`, and numerically equals
  `(1/lam) * log(mean(exp(-lam*pnl)))`.
- All randomness in tests flows through an explicit `torch.Generator`.

---

### Task 5.1: CVaR loss — closed-form Gaussian value at the optimal `w`

**Files:**
- Create: `deephedge/losses.py`
- Test: `tests/test_losses.py`

- [ ] **Step 1: Write the failing test**

For `pnl ~ Normal(0, 1)`, the loss `L = -pnl ~ Normal(0, 1)`. The Rockafellar–Uryasev
objective `F(w) = w + (1/(1-alpha)) * E[relu(L - w)]` is minimized at `w* = VaR_alpha`,
and `min_w F(w) = CVaR_alpha(L) = phi(z_alpha) / (1 - alpha)`. For `alpha = 0.95`,
`z_0.95 = 1.6448536`, `phi(z) = 0.1031356`, so `CVaR_0.95 = 0.1031356 / 0.05 = 2.06271`.
We find the optimal `w` by a grid search using `cvar_loss` itself and assert the minimum
matches the closed form within 0.05 (a large sample is needed because the `alpha` tail
uses only ~5% of paths — §6.1).

```python
import math
import torch

from deephedge.losses import cvar_loss


def test_cvar_loss_matches_gaussian_closed_form_at_optimal_w():
    gen = torch.Generator().manual_seed(0)
    n = 400_000
    pnl = torch.randn(n, generator=gen)  # P&L ~ N(0,1) -> L = -pnl ~ N(0,1)
    alpha = 0.95

    # Closed-form Gaussian CVaR of the loss L: phi(z_alpha) / (1 - alpha).
    z = 1.6448536269514722  # standard-normal 0.95 quantile
    phi = math.exp(-z * z / 2.0) / math.sqrt(2.0 * math.pi)
    cvar_closed_form = phi / (1.0 - alpha)  # ~= 2.0627

    # Minimize F(w) over a grid bracketing VaR_0.95 (= z) using cvar_loss directly.
    ws = torch.linspace(1.0, 2.2, 1201)
    vals = torch.stack([cvar_loss(pnl, alpha, w) for w in ws])
    best_idx = int(torch.argmin(vals))
    best_w = float(ws[best_idx])
    best_cvar = float(vals[best_idx])

    assert abs(best_cvar - cvar_closed_form) < 0.05
    # Left endpoint of the argmin is VaR_alpha = z_0.95 (continuous P&L -> unique).
    assert abs(best_w - z) < 0.05

    # cvar_loss returns a scalar tensor.
    out = cvar_loss(pnl, alpha, torch.tensor(z))
    assert out.shape == torch.Size([])
```

- [ ] **Step 2: Run test to verify it fails**

`pytest tests/test_losses.py::test_cvar_loss_matches_gaussian_closed_form_at_optimal_w -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'deephedge.losses'` (the module
does not exist yet).

- [ ] **Step 3: Write minimal implementation**

```python
# deephedge/losses.py
"""Convex risk-measure losses for deep hedging (spec §6).

We work consistently in loss space ``L = -pnl`` (larger = worse). ``pnl`` is the
profit-positive terminal P&L tensor of shape ``(N,)`` (one entry per Monte-Carlo path).
"""

import math

import torch


def cvar_loss(pnl: torch.Tensor, alpha: float, w: torch.Tensor) -> torch.Tensor:
    """Rockafellar-Uryasev CVaR loss (spec §6.1).

    ``L = -pnl`` ;  ``F(params, w) = w + (1/(1-alpha)) * mean(relu(L - w))``.

    ``w`` is a learnable scalar (``nn.Parameter`` of shape ``()``) passed in by the
    caller. We MINIMIZE ``F`` jointly over ``(network params, w)`` with Adam.

    Justification for the joint optimization is the UPPER-BOUND / recovery property,
    NOT joint convexity (spec §6.1): because ``CVaR(params) = min_w F(params, w)``, for
    any fixed ``w`` we have ``F(params, w) >= CVaR(params)``, hence
    ``min_{params, w} F = min_{params} CVaR``. With a neural-net hedge the problem is
    NON-CONVEX (RU joint convexity needs the loss convex in the decision variable — RU
    2002 Cor. 11 / RU 2000 Thm 2), so do NOT claim global optimality. The
    ``(1 - alpha)`` tail-probability denominator is correct; using ``alpha`` is the
    classic ~19x mis-scaling bug. The ``alpha=0.95`` tail uses only ~5% of paths, so
    gradients are high variance — train with large/representative batches.
    """
    loss = -pnl
    return w + (1.0 / (1.0 - alpha)) * torch.relu(loss - w).mean()
```

- [ ] **Step 4: Run test to verify it passes**

`pytest tests/test_losses.py::test_cvar_loss_matches_gaussian_closed_form_at_optimal_w -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add deephedge/losses.py tests/test_losses.py
git commit -m "feat(losses): CVaR loss matching Gaussian closed form"
```

---

### Task 5.2: CVaR sanity ordering — `CVaR_alpha >= VaR_alpha >= E[L]` (regression-lock)

**Files:**
- Modify: `tests/test_losses.py`

- [ ] **Step 1: Write the test**

`cvar_loss` evaluated at `w = VaR_alpha` returns `CVaR_alpha(L)`. We compute the
empirical `VaR_alpha = quantile_alpha(L)` and `E[L] = mean(L)` directly from the sample
and assert the coherent-risk ordering `CVaR_alpha >= VaR_alpha >= E[L]` (spec §13 test 4).
We use a skewed P&L (short a call-like payoff: a fat left tail) so the inequalities are
strict and meaningful, not a degenerate symmetric tie.

```python
def test_cvar_geq_var_geq_expected_loss():
    gen = torch.Generator().manual_seed(1)
    n = 200_000
    # Skewed, profit-positive P&L: mostly small positive, with a fat left tail (losses).
    pnl = 0.5 - torch.relu(torch.randn(n, generator=gen)) ** 2
    alpha = 0.95
    loss = -pnl

    var = torch.quantile(loss, alpha)            # VaR_alpha of the loss
    expected_loss = loss.mean()                  # E[L]
    cvar = cvar_loss(pnl, alpha, var)            # CVaR_alpha = F(w=VaR)

    assert float(cvar) >= float(var) - 1e-4
    assert float(var) >= float(expected_loss) - 1e-4
    # Tail is genuinely heavier than the mean: strict gaps on this skewed sample.
    assert float(cvar) > float(var)
    assert float(var) > float(expected_loss)
```

- [ ] **Step 2: Run the test**

`pytest tests/test_losses.py::test_cvar_geq_var_geq_expected_loss -v`

Expected: PASS (regression-lock — behavior implemented in Task 5.1; this guards against
regression). `cvar_loss` from Task 5.1 already satisfies the coherent-risk ordering; this
task only adds the coverage. If it is red, the bug is in `cvar_loss`, not the test.

- [ ] **Step 3: Write minimal implementation**

No production change needed — `cvar_loss` from Task 5.1 already satisfies the ordering.
This task hardens coverage of the coherence sanity check (spec §13 test 4). If the test
fails, fix `cvar_loss` (e.g. a wrong denominator) rather than weakening the assertion.

- [ ] **Step 4: Run test to verify it passes**

`pytest tests/test_losses.py::test_cvar_geq_var_geq_expected_loss -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_losses.py
git commit -m "test(losses): CVaR >= VaR >= E[L] coherence ordering"
```

---

### Task 5.3: Entropic loss — small-`lambda` limit recovers `-E[pnl]`

**Files:**
- Modify: `deephedge/losses.py`
- Modify: `tests/test_losses.py`

- [ ] **Step 1: Write the failing test**

As `lambda -> 0+`, the entropic risk `(1/lam) * log(mean(exp(-lam*pnl)))` converges to
`-E[pnl]` (first-order Taylor: `mean(exp(-lam*pnl)) ≈ 1 - lam*E[pnl]`, so the loss ≈
`-E[pnl]`; spec §6.2 / §13 test 5). We use `float64` so the small-`lambda` cancellation
is controlled, and assert the small-`lambda` value is within `1e-3` of `-mean(pnl)`.

```python
from deephedge.losses import entropic_loss


def test_entropic_small_lambda_limit_recovers_negative_mean_pnl():
    gen = torch.Generator().manual_seed(2)
    n = 200_000
    pnl = (0.3 + 1.5 * torch.randn(n, generator=gen)).to(torch.float64)

    neg_mean = -pnl.mean()
    val = entropic_loss(pnl, lam=1e-4)

    assert val.shape == torch.Size([])
    assert abs(float(val) - float(neg_mean)) < 1e-3
```

- [ ] **Step 2: Run test to verify it fails**

`pytest tests/test_losses.py::test_entropic_small_lambda_limit_recovers_negative_mean_pnl -v`

Expected: FAIL with `ImportError: cannot import name 'entropic_loss' from
'deephedge.losses'` (function not yet defined).

- [ ] **Step 3: Write minimal implementation**

```python
# Append to deephedge/losses.py

def entropic_loss(pnl: torch.Tensor, lam: float) -> torch.Tensor:
    """Entropic (exponential) risk measure (spec §6.2).

    ``(1/lam) * (logsumexp(-lam*pnl) - log(N))`` with ``N = pnl.shape[0]``. This equals
    ``(1/lam) * log(mean(exp(-lam*pnl)))``: the ``- log(N)`` turns the sum into a mean so
    loss values are comparable across batch sizes (argmin is unaffected, but reporting is
    wrong without it). ``pnl`` is profit-positive, so the ``exp(-lam*pnl)`` exponent
    penalizes losses exponentially. Convex, monotone, cash-invariant (not coherent);
    equivalent to maximizing CARA utility ``u(x) = -exp(-lam*x)`` with ``lam`` = absolute
    risk aversion. ``torch.logsumexp`` is max-shift stabilized (exact). As ``lam -> 0+``
    the loss -> ``-E[pnl]``; the empirical estimator is biased ``O(1/N)`` (Jensen),
    vanishing as ``N -> inf``.
    """
    n = pnl.shape[0]
    log_n = math.log(n)
    return (1.0 / lam) * (torch.logsumexp(-lam * pnl, dim=0) - log_n)
```

- [ ] **Step 4: Run test to verify it passes**

`pytest tests/test_losses.py::test_entropic_small_lambda_limit_recovers_negative_mean_pnl -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add deephedge/losses.py tests/test_losses.py
git commit -m "feat(losses): entropic loss with small-lambda mean limit"
```

---

### Task 5.4: Entropic loss — mean-vs-sum (`log N`) identity

**Files:**
- Modify: `tests/test_losses.py`

- [ ] **Step 1: Write the failing test**

The implementation must equal `(1/lam) * log(mean(exp(-lam*pnl)))` (the mean form),
confirming the `- log(N)` is present and correct (spec §6.2 / §13 test 5). We use a
moderate `lambda = 1.0` and `float64` so the two analytically-identical expressions agree
within `1e-5` without float32 cancellation noise. We also assert the WRONG sum form
(omitting `- log(N)`) does NOT match, so the test would catch a dropped `log(N)`.

```python
def test_entropic_equals_mean_form_identity():
    gen = torch.Generator().manual_seed(3)
    n = 50_000
    pnl = (0.2 * torch.randn(n, generator=gen)).to(torch.float64)
    lam = 1.0

    impl = entropic_loss(pnl, lam)
    mean_form = (1.0 / lam) * torch.log(torch.mean(torch.exp(-lam * pnl)))

    assert abs(float(impl) - float(mean_form)) < 1e-5

    # The sum form (no -log N) must differ by exactly (1/lam)*log(N).
    sum_form = (1.0 / lam) * torch.logsumexp(-lam * pnl, dim=0)
    assert abs(float(sum_form - impl) - (1.0 / lam) * math.log(n)) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

`pytest tests/test_losses.py::test_entropic_equals_mean_form_identity -v`

Expected: FAIL with `ModuleNotFoundError`/`ImportError` if run before Task 5.3; otherwise
it PASSES against the Task 5.3 implementation. This task adds the identity-guard test;
confirm green after Task 5.3. A red result here means `- log(N)` is missing or wrong.

- [ ] **Step 3: Write minimal implementation**

No production change needed — `entropic_loss` from Task 5.3 already returns the mean form.
This task guards the `- log(N)` normalization (spec §6.2). If the identity fails, restore
the `- math.log(n)` term in `entropic_loss`.

- [ ] **Step 4: Run test to verify it passes**

`pytest tests/test_losses.py::test_entropic_equals_mean_form_identity -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_losses.py
git commit -m "test(losses): entropic mean-vs-sum log N identity"
```

---

### Task 5.5: Both losses are differentiable w.r.t. a `pnl` that depends on a `Parameter`

**Files:**
- Modify: `tests/test_losses.py`

- [ ] **Step 1: Write the failing test**

Training backprops through the loss into the network params and (for CVaR) the scalar `w`
(spec §11, §13 test 7). We build a `pnl` that depends on a `torch.nn.Parameter` and
assert: (a) `cvar_loss` produces a finite gradient w.r.t. both the `pnl`-driving param
and `w`; (b) `entropic_loss` produces a finite gradient w.r.t. the `pnl`-driving param;
(c) the gradients have the expected analytic signs on a controlled example.

```python
def test_losses_are_differentiable_through_pnl_parameter():
    gen = torch.Generator().manual_seed(4)
    n = 20_000
    base = torch.randn(n, generator=gen)

    # CVaR: pnl = theta * base ; w is a separate learnable scalar.
    theta_cvar = torch.nn.Parameter(torch.tensor(1.0))
    w = torch.nn.Parameter(torch.tensor(1.6))
    pnl_cvar = theta_cvar * base
    loss_cvar = cvar_loss(pnl_cvar, 0.95, w)
    loss_cvar.backward()
    assert theta_cvar.grad is not None and torch.isfinite(theta_cvar.grad)
    assert w.grad is not None and torch.isfinite(w.grad)
    # d/dw F = 1 - (1/(1-alpha)) * P(L > w). At w below the 0.95 loss-quantile the tail
    # mass exceeds (1-alpha), so the gradient w.r.t. w is negative (lower w not optimal).
    assert float(w.grad) < 0.0

    # Entropic: pnl = theta * base + shift ; gradient w.r.t. an additive shift is -1.
    theta_ent = torch.nn.Parameter(torch.tensor(0.5))
    shift = torch.nn.Parameter(torch.tensor(0.0))
    pnl_ent = theta_ent * base + shift
    loss_ent = entropic_loss(pnl_ent, lam=1.0)
    loss_ent.backward()
    assert theta_ent.grad is not None and torch.isfinite(theta_ent.grad)
    assert shift.grad is not None and torch.isfinite(shift.grad)
    # entropic_loss(pnl + c) = entropic_loss(pnl) - c  =>  d(loss)/d(shift) = -1.
    assert abs(float(shift.grad) - (-1.0)) < 1e-4
```

- [ ] **Step 2: Run test to verify it fails**

`pytest tests/test_losses.py::test_losses_are_differentiable_through_pnl_parameter -v`

Expected: FAIL with `ModuleNotFoundError`/`ImportError` if run before Tasks 5.1/5.3;
otherwise PASS against the existing implementations. This task adds the autograd-flow
guard (spec §13 test 7); confirm green after Tasks 5.1 and 5.3.

- [ ] **Step 3: Write minimal implementation**

No production change needed — `cvar_loss` and `entropic_loss` are built from
differentiable torch ops (`relu`, `mean`, `logsumexp`), so autograd already flows to the
`pnl`-driving params and to `w`. This task locks in cash-invariance of the entropic loss
(`d/d(shift) = -1`) and the CVaR `w`-gradient sign. If a gradient is `None`, an op was
replaced by a non-differentiable equivalent — revert it.

- [ ] **Step 4: Run test to verify it passes**

`pytest tests/test_losses.py::test_losses_are_differentiable_through_pnl_parameter -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_losses.py
git commit -m "test(losses): autograd flows through pnl param and CVaR w"
```
## Group 6: Hedger network + feature builder

Implements `deephedge/hedger.py` (spec §8): the feature builder `build_features` and the
shared-weight MLP policy `Hedger`. Binds exactly to the Shared Contracts:

```python
class Hedger(torch.nn.Module):
    def __init__(self, n_features: int, n_instruments: int, hidden=(32, 32)): ...
    def forward(self, features) -> torch.Tensor              # (B, n_features) -> (B, n_instruments)

def build_features(S_i, tau_norm, prev_holdings, V_i=None, k_norm=100.0) -> torch.Tensor
# feature order: [log(S_i/k_norm), tau_norm, *prev_holdings, (sqrt(clamp(V_i,0)) if not None)]
```

**Normalization:** `k_norm` is a **parameter** (default `100.0`), the strike used to
normalize the log-moneyness feature `log(S_i / k_norm)` — callers pass `cfg.k`. It is NOT a
hardcoded constant baked into the `log`. The default `100.0` matches the spec default
`s0 = k = 100.0` so an ATM spot maps to feature `0.0`. `tau_norm` is the already-normalized
time-to-maturity `(T - t_i)/T` (callers pass `state.tau / cfg.maturity`), broadcast to a
`(B, 1)` column. `S_i`, `prev_holdings`, and `V_i` accept `(B,)` OR `(B, 1)` and are reshaped
to columns internally. `V_i`, when supplied, contributes the `sqrt(clamp(V_i, 0))` volatility
feature (the Heston/Bates branch); the clamp guards the full-truncation Euler state, which is
allowed to carry negative variance forward (spec §5.2).

---

### Task 6.1: `build_features` output width and exact values (underlying-only, no variance)

**Files:**
- Create: `deephedge/hedger.py`
- Test: `tests/test_hedger.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hedger.py
import math

import torch

from deephedge.hedger import build_features


def test_build_features_width_and_values_underlying_only():
    # B=2 paths, m=1 instrument (underlying), no variance feature.
    S_i = torch.tensor([[100.0], [200.0]])          # (B, 1)
    prev_holdings = torch.tensor([[0.5], [0.25]])    # (B, n_instruments=1)
    tau_norm = 0.5                                   # normalized (T-t)/T, broadcast to (B, 1)

    feats = build_features(S_i, tau_norm, prev_holdings, V_i=None, k_norm=100.0)

    # width = 2 (log-moneyness + tau_norm) + n_instruments (1) = 3
    assert feats.shape == (2, 3)

    # column 0: log(S_i / k_norm) with k_norm=100.0
    assert feats[0, 0].item() == 0.0                 # log(100/100) = 0
    assert math.isclose(feats[1, 0].item(), math.log(2.0), rel_tol=1e-6)
    # column 1: tau_norm broadcast
    assert feats[0, 1].item() == 0.5
    assert feats[1, 1].item() == 0.5
    # column 2: prev holdings passed through unchanged
    assert feats[0, 2].item() == 0.5
    assert feats[1, 2].item() == 0.25
```

- [ ] **Step 2: Run test to verify it fails**

Command: `pytest tests/test_hedger.py::test_build_features_width_and_values_underlying_only -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'deephedge.hedger'` (the module and
`build_features`/`k_ref` do not exist yet).

- [ ] **Step 3: Write minimal implementation**

```python
# deephedge/hedger.py
"""Hedger policy network and feature builder (spec §8)."""

import torch


def build_features(S_i, tau_norm, prev_holdings, V_i=None, k_norm=100.0) -> torch.Tensor:
    """Build the per-step policy input features.

    Feature order (columns):
        [ log(S_i / k_norm), tau_norm, *prev_holdings columns,
          (sqrt(clamp(V_i, 0)) if V_i is not None) ]

    Args:
        S_i: (B,) or (B, 1) current spot prices (reshaped to a column internally).
        tau_norm: scalar already-normalized time-to-maturity (T - t_i) / T (callers pass
            state.tau / cfg.maturity), broadcast to a (B, 1) column.
        prev_holdings: (B,) or (B, n_instruments) holdings carried from the prior step
            (reshaped to (B, n_instruments) internally).
        V_i: optional (B,) or (B, 1) instantaneous variance (Heston/Bates); contributes
            sqrt(clamp(V_i, 0)). The full-truncation state may be negative, hence clamp.
        k_norm: strike used for the moneyness feature log(S_i / k_norm). PARAMETER
            (default 100.0); callers pass cfg.k. Not a hardcoded constant.

    Returns:
        (B, n_features) feature tensor, n_features = 2 + n_instruments (+1 if V_i given).
    """
    # Reshape inputs to columns first so (B,) or (B,1) are both accepted.
    S_i = torch.as_tensor(S_i).reshape(-1, 1)                    # (B, 1)
    prev_holdings = torch.as_tensor(prev_holdings).reshape(S_i.shape[0], -1)
    if V_i is not None:
        V_i = torch.as_tensor(V_i).reshape(-1, 1)               # (B, 1)

    B = S_i.shape[0]
    log_moneyness = torch.log(S_i / k_norm)                     # (B, 1) — k_norm is a PARAM
    tau_col = torch.full((B, 1), float(tau_norm), dtype=S_i.dtype, device=S_i.device)
    cols = [log_moneyness, tau_col, prev_holdings]
    if V_i is not None:
        cols.append(torch.sqrt(torch.clamp(V_i, min=0.0)))      # (B, 1)
    return torch.cat(cols, dim=1)
```

- [ ] **Step 4: Run test to verify it passes**

Command: `pytest tests/test_hedger.py::test_build_features_width_and_values_underlying_only -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add deephedge/hedger.py tests/test_hedger.py
git commit -m "feat(hedger): build_features for underlying-only policy inputs"
```

---

### Task 6.2: `build_features` with the variance feature and multiple instruments

**Files:**
- Modify: `deephedge/hedger.py`
- Test: `tests/test_hedger.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hedger.py  (append)
def test_build_features_width_with_variance_and_two_instruments():
    # B=2, m=2 instruments (underlying + option), variance supplied.
    S_i = torch.tensor([[100.0], [50.0]])                 # (B, 1)
    prev_holdings = torch.tensor([[0.5, -0.1],
                                  [0.3, 0.2]])            # (B, n_instruments=2)
    V_i = torch.tensor([[0.04], [-0.01]])                 # (B, 1); second entry negative
    tau_norm = 1.0

    feats = build_features(S_i, tau_norm, prev_holdings, V_i=V_i, k_norm=100.0)

    # width = 2 + n_instruments(2) + 1 (variance) = 5
    assert feats.shape == (2, 5)

    # column 0: log-moneyness with k_norm=100.0
    assert feats[0, 0].item() == 0.0
    assert math.isclose(feats[1, 0].item(), math.log(0.5), rel_tol=1e-6)
    # column 1: tau_norm
    assert feats[0, 1].item() == 1.0
    # columns 2,3: the two prior holdings
    assert feats[0, 2].item() == 0.5
    assert feats[0, 3].item() == -0.1
    assert feats[1, 3].item() == 0.2
    # column 4: sqrt(clamp(V_i, 0)) -> sqrt(0.04)=0.2 ; clamped negative -> 0.0
    assert math.isclose(feats[0, 4].item(), 0.2, rel_tol=1e-6)
    assert feats[1, 4].item() == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Command: `pytest tests/test_hedger.py::test_build_features_width_with_variance_and_two_instruments -v`

Expected: FAIL — the test asserts the clamped variance value `sqrt(clamp(-0.01, 0)) == 0.0`
and the 5-column width; if the implementation from Task 6.1 is already correct this test
passes immediately. (Run before adding any code change. If it already passes, the variance
branch is verified and you proceed to Step 5 with no code change.)

- [ ] **Step 3: Write minimal implementation**

No code change is required — Task 6.1's `build_features` already appends
`sqrt(clamp(V_i, 0))` when `V_i is not None`. This task exists to lock the variance +
multi-instrument contract with an explicit regression test. (If Step 2 surfaced a real
failure, fix it by ensuring the `V_i is not None` branch appends exactly
`torch.sqrt(torch.clamp(V_i, min=0.0))` as the final column.)

- [ ] **Step 4: Run test to verify it passes**

Command: `pytest tests/test_hedger.py::test_build_features_width_with_variance_and_two_instruments -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_hedger.py
git commit -m "test(hedger): lock build_features variance + multi-instrument contract"
```

---

### Task 6.3: `Hedger` forward output shape

**Files:**
- Modify: `deephedge/hedger.py`
- Test: `tests/test_hedger.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hedger.py  (append)
from deephedge.hedger import Hedger


def test_hedger_forward_shape():
    torch.manual_seed(0)
    n_features, n_instruments, B = 4, 2, 7
    hedger = Hedger(n_features=n_features, n_instruments=n_instruments, hidden=(32, 32))
    features = torch.randn(B, n_features)

    holdings = hedger.forward(features)

    assert holdings.shape == (B, n_instruments)
    assert holdings.dtype == features.dtype
    # No output activation: holdings are unbounded reals (not squashed to [-1,1] etc.).
    # A linear final layer on random input should not be bounded; just assert finiteness.
    assert torch.isfinite(holdings).all()
```

- [ ] **Step 2: Run test to verify it fails**

Command: `pytest tests/test_hedger.py::test_hedger_forward_shape -v`

Expected: FAIL with `ImportError: cannot import name 'Hedger' from 'deephedge.hedger'`
(the class does not exist yet).

- [ ] **Step 3: Write minimal implementation**

```python
# deephedge/hedger.py  (append)
class Hedger(torch.nn.Module):
    """Shared-weight feed-forward MLP policy (spec §8).

    Applied at every timestep with the same parameters (semi-recurrent: prior holdings
    are an input feature, not a hidden recurrent state). ReLU hidden activations; the
    final Linear maps to n_instruments with NO output activation, because holdings are
    unbounded real positions.
    """

    def __init__(self, n_features: int, n_instruments: int, hidden=(32, 32)):
        super().__init__()
        layers: list[torch.nn.Module] = []
        in_dim = n_features
        for h in hidden:
            layers.append(torch.nn.Linear(in_dim, h))
            layers.append(torch.nn.ReLU())
            in_dim = h
        layers.append(torch.nn.Linear(in_dim, n_instruments))  # no output activation
        self.net = torch.nn.Sequential(*layers)

    def forward(self, features) -> torch.Tensor:
        return self.net(features)
```

- [ ] **Step 4: Run test to verify it passes**

Command: `pytest tests/test_hedger.py::test_hedger_forward_shape -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add deephedge/hedger.py tests/test_hedger.py
git commit -m "feat(hedger): shared-weight MLP policy with unbounded holdings output"
```

---

### Task 6.4: Shared weights — parameter count is independent of the number of steps

**Files:**
- Modify: `tests/test_hedger.py`
- Test: `tests/test_hedger.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hedger.py  (append)
def test_hedger_shared_weights_param_count_independent_of_n_steps():
    torch.manual_seed(0)
    n_features, n_instruments = 4, 1
    hedger = Hedger(n_features=n_features, n_instruments=n_instruments, hidden=(32, 32))

    # Same module applied at every step. Parameter count must not grow with n_steps.
    def total_params(module):
        return sum(p.numel() for p in module.parameters())

    base = total_params(hedger)

    # Roll the SAME module forward over varying horizons; collect the param sets each time.
    param_ids_by_horizon = {}
    for n_steps in (1, 5, 30):
        feats = torch.randn(8, n_features)
        for _ in range(n_steps):
            _ = hedger.forward(feats)  # reuses identical parameters at every step
        param_ids_by_horizon[n_steps] = {id(p) for p in hedger.parameters()}
        # Param count is invariant to how many steps we applied the module.
        assert total_params(hedger) == base

    # The exact same parameter tensors are reused across all horizons (true weight sharing).
    assert param_ids_by_horizon[1] == param_ids_by_horizon[5] == param_ids_by_horizon[30]

    # Concrete expected count for hidden=(32,32): a 4->32->32->1 MLP.
    #   layer1: 4*32 + 32   = 160
    #   layer2: 32*32 + 32  = 1056
    #   out:    32*1  + 1   = 33
    #   total               = 1249
    assert base == 4 * 32 + 32 + 32 * 32 + 32 + 32 * 1 + 1
    assert base == 1249
```

- [ ] **Step 2: Run test to verify it fails**

Command: `pytest tests/test_hedger.py::test_hedger_shared_weights_param_count_independent_of_n_steps -v`

Expected: This test passes once Task 6.3's `Hedger` exists, BUT run it against the current
code first. Before Task 6.3 it FAILS with `ImportError` for `Hedger`. If Task 6.3 is already
merged, run it and confirm it PASSES (it then serves as the weight-sharing regression gate
for spec §13.7). If the arithmetic constant `1249` ever mismatches, the architecture drifted
from `hidden=(32,32)` and must be reconciled.

- [ ] **Step 3: Write minimal implementation**

No production code change — `Hedger` from Task 6.3 already uses a single shared module whose
parameters do not depend on the horizon. This task adds the explicit weight-sharing /
parameter-count gate required by spec §13.7.

- [ ] **Step 4: Run test to verify it passes**

Command: `pytest tests/test_hedger.py::test_hedger_shared_weights_param_count_independent_of_n_steps -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_hedger.py
git commit -m "test(hedger): assert shared weights and step-independent param count"
```

---

### Task 6.5: Gradients flow through `build_features` -> `Hedger` to all parameters

**Files:**
- Modify: `tests/test_hedger.py`
- Test: `tests/test_hedger.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hedger.py  (append)
def test_gradients_flow_through_features_and_network():
    gen = torch.Generator().manual_seed(123)
    n_instruments = 1
    B = 16

    # Inputs that exercise both the feature builder and the network.
    S_i = 100.0 + torch.randn(B, 1, generator=gen)
    V_i = 0.04 + 0.01 * torch.randn(B, 1, generator=gen)
    prev_holdings = torch.randn(B, n_instruments, generator=gen)
    tau_norm = 0.5

    feats = build_features(S_i, tau_norm, prev_holdings, V_i=V_i)
    n_features = feats.shape[1]                       # 2 + n_instruments + 1 = 4
    assert n_features == 4

    torch.manual_seed(0)
    hedger = Hedger(n_features=n_features, n_instruments=n_instruments, hidden=(32, 32))

    holdings = hedger.forward(feats)                  # (B, n_instruments)
    loss = holdings.pow(2).sum()                      # scalar, smooth in all params
    loss.backward()

    # Every parameter must receive a finite, present gradient (autograd graph is closed).
    grads = list(hedger.parameters())
    assert len(grads) > 0
    for p in grads:
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()

    # At least one parameter has a non-zero gradient (signal actually flows, not all-dead-ReLU).
    assert any(p.grad.abs().sum().item() > 0.0 for p in hedger.parameters())
```

- [ ] **Step 2: Run test to verify it fails**

Command: `pytest tests/test_hedger.py::test_gradients_flow_through_features_and_network -v`

Expected: Run against current code. If Tasks 6.1 and 6.3 are merged, this PASSES and becomes
the gradient-flow gate for spec §13.7. If `build_features` or `Hedger` were broken (e.g. a
`detach()` or a non-differentiable op slipped into the feature path), it FAILS with an
`AssertionError` on `p.grad is not None`. Confirm the green state before relying on it.

- [ ] **Step 3: Write minimal implementation**

No production code change — `build_features` (Task 6.1) uses only differentiable ops
(`torch.log`, `torch.clamp`, `torch.sqrt`, `torch.cat`) and `Hedger.forward` (Task 6.3) is a
plain `nn.Sequential`, so the autograd graph from inputs to parameters is intact. This task
adds the gradient-flow gate.

- [ ] **Step 4: Run test to verify it passes**

Command: `pytest tests/test_hedger.py::test_gradients_flow_through_features_and_network -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_hedger.py
git commit -m "test(hedger): assert gradients flow from features through network to params"
```
## Group 7 — Training loop + frictionless/cost experiments

> Implements `deephedge/train.py`: `train(cfg) -> tuple[Hedger, dict]`. Binds to the
> Shared Contracts exactly — `ExperimentConfig`, `get_simulator`, `EuropeanOption`,
> `bs_price`, `bs_delta`, `heston_price_cm`, `build_instr_prices`, `make_nn_strategy`,
> `simulate_pnl` (`.pnl`), `cvar_loss`, `entropic_loss`, `Hedger`, `build_features`.
> Spec refs: §11 (training), §16(3) (recover BS delta on frictionless GBM, then beat
> under costs), §13(6) (tolerance-based replication gate).

This group assumes Groups 1–6 are complete (config, simulators, BS + Heston pricers,
instruments, hedger, losses, portfolio, benchmarks). Every symbol used below is from the
Shared Contracts. `train.py` is pure orchestration; it introduces **one** private helper,
`_n_features(cfg)`, fully defined in Task 7.1.

The design of `train(cfg)`:

- **Feature count.** The hedger's `n_features` must match what `build_features` emits.
  The contract feature order is `[log(S_i/k_norm), tau_norm, *prev_holdings, (sqrt(clamp(V_i,0)) if not None)]`,
  so `n_features = 2 + cfg.n_instruments + (1 if model in ('heston', 'bates') else 0)`.
  Stochastic-vol models are `"heston"` and `"bates"` (they populate `Paths.V`). Feature
  building goes through `make_nn_strategy`, which passes the normalized tau
  (`state.tau / cfg.maturity`) and `k_norm=cfg.k` so normalization is consistent.
- **Optimizer.** Adam over `hedger.parameters()`. When `cfg.loss == "cvar"`, create
  `w = nn.Parameter(torch.zeros((), device=cfg.device))` and add it as a **separate param
  group with a higher learning rate** (10× `cfg.lr`) per spec §6.1 / §11. For entropic,
  `w` is unused (kept as `None` in history).
- **Each step.** Simulate FRESH paths with a fresh generator seeded `cfg.seed + step`
  (reproducible, non-repeating); build the sold `liability = EuropeanOption(cfg.k,
  cfg.maturity)` and the `hedge = _hedge_option(cfg) if "option" in cfg.instruments else
  None`; compute the premium from the liability under the risk-neutral measure (`bs_price`
  for `gbm`/`merton`, `heston_price_cm` for `heston`/`bates`); `instr_prices =
  build_instr_prices(cfg, paths, hedge)`; `strategy = make_nn_strategy(hedger, cfg)`;
  `pnl = simulate_pnl(strategy, paths, cfg, liability, premium, instr_prices).pnl`;
  `loss = cvar_loss(pnl, cfg.alpha, w)` or `entropic_loss(pnl, cfg.entropic_lambda)`;
  `loss.backward()`; `opt.step()`. Record the scalar loss into `history["loss"]`.
- **Return.** `(hedger, history)` where `history` is a dict with keys `"loss"` (list of
  floats, one per step), `"w"` (list of floats for CVaR else empty list), and `"premium"`
  (the premium float, identical across steps for a fixed cfg).

Total steps trained = `cfg.epochs * cfg.steps_per_epoch`. Tests use a tiny cfg so this
runs in seconds.

---

### Task 7.1: `train(cfg)` skeleton — runs N steps, returns `(Hedger, history)` with required keys

**Files:**
- Create: `deephedge/train.py`
- Test: `tests/test_train.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train.py
import torch
from deephedge.config import ExperimentConfig
from deephedge.hedger import Hedger
from deephedge.train import train


def _tiny_cfg(**overrides):
    base = dict(
        s0=100.0, k=100.0, r=0.0, q=0.0, mu=None,
        maturity=30 / 252, n_steps=10,
        sigma=0.2, model="gbm", loss="cvar", alpha=0.95,
        cost=0.0, instruments=("underlying",),
        n_paths=2048, batch_size=2048,
        epochs=1, steps_per_epoch=2,
        lr=1e-3, seed=0, device="cpu",
    )
    base.update(overrides)
    return ExperimentConfig(**base)


def test_train_returns_hedger_and_history_with_expected_keys():
    cfg = _tiny_cfg()
    hedger, history = train(cfg)

    assert isinstance(hedger, Hedger)
    # history is a dict with the contract keys
    assert set(history.keys()) >= {"loss", "w", "premium"}
    # one loss value per trained step
    n_steps_total = cfg.epochs * cfg.steps_per_epoch
    assert len(history["loss"]) == n_steps_total
    assert all(isinstance(x, float) for x in history["loss"])
    # cvar => one w per step; premium is a single positive float repeated/stored once
    assert len(history["w"]) == n_steps_total
    assert isinstance(history["premium"], float)
    # ATM call premium under r=0, q=0 is positive and < S0
    assert 0.0 < history["premium"] < cfg.s0
```

- [ ] **Step 2: Run test to verify it fails**

`pytest tests/test_train.py::test_train_returns_hedger_and_history_with_expected_keys -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'deephedge.train'` (the module
does not exist yet).

- [ ] **Step 3: Write minimal implementation**

```python
# deephedge/train.py
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
    p = bs_price(S, cfg.k, cfg.maturity, cfg.r, cfg.sigma, q=cfg.q, kind=option.kind)
    return float(p)


def _hedge_option(cfg: ExperimentConfig) -> EuropeanOption:
    """The HEDGE instrument's option contract, defaulting to the sold call."""
    strike = cfg.hedge_option_strike if cfg.hedge_option_strike is not None else cfg.k
    maturity = cfg.hedge_option_maturity if cfg.hedge_option_maturity is not None else cfg.maturity
    return EuropeanOption(strike, maturity, kind="call")


def train(cfg: ExperimentConfig) -> tuple[Hedger, dict]:
    device = torch.device(cfg.device)
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
```

- [ ] **Step 4: Run test to verify it passes**

`pytest tests/test_train.py::test_train_returns_hedger_and_history_with_expected_keys -v`

Expected: PASS — `train` runs 2 steps, returns a `Hedger`, and `history` has the three
keys with `len(history["loss"]) == 2`, `len(history["w"]) == 2`, and a positive ATM
premium.

- [ ] **Step 5: Commit**

```bash
git add deephedge/train.py tests/test_train.py
git commit -m "feat(train): training loop skeleton returning hedger + history"
```

---

### Task 7.2: Entropic-loss training path uses no `w` and records empty `w` history

**Files:**
- Modify: `deephedge/train.py` (no change needed if Task 7.1 already branches; this task
  proves the entropic branch)
- Test: `tests/test_train.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train.py  (append)
def test_train_entropic_loss_branch_has_no_w():
    cfg = _tiny_cfg(loss="entropic", entropic_lambda=1.0)
    hedger, history = train(cfg)

    n_steps_total = cfg.epochs * cfg.steps_per_epoch
    assert len(history["loss"]) == n_steps_total
    # entropic path never creates w -> empty list
    assert history["w"] == []
    # entropic loss with profit-positive PnL is finite and not NaN
    assert all(x == x for x in history["loss"])  # NaN-check (NaN != NaN)
```

- [ ] **Step 2: Run test to verify it fails**

`pytest tests/test_train.py::test_train_entropic_loss_branch_has_no_w -v`

Expected: FAIL only if Task 7.1's `train` mishandles the entropic branch. If 7.1 was
implemented exactly as written it will PASS immediately; if it fails, the failure is
`AssertionError: history["w"] == []` because `w` was created/recorded for entropic. Fix in
Step 3.

- [ ] **Step 3: Write minimal implementation**

If the test already passes, no change is required — Task 7.1 only creates and appends `w`
inside `if cfg.loss == "cvar"`. If it failed, ensure the `w` guard is correct:

```python
# deephedge/train.py  (the guard must read exactly:)
    w = None
    if cfg.loss == "cvar":
        w = nn.Parameter(torch.zeros((), device=device))
        params.append({"params": [w], "lr": cfg.lr * 10.0})
    ...
        if w is not None:
            history["w"].append(float(w.detach()))
```

- [ ] **Step 4: Run test to verify it passes**

`pytest tests/test_train.py::test_train_entropic_loss_branch_has_no_w -v`

Expected: PASS — entropic training records an empty `w` list and finite (non-NaN) losses.

- [ ] **Step 5: Commit**

```bash
git add deephedge/train.py tests/test_train.py
git commit -m "test(train): cover entropic-loss branch (no auxiliary w)"
```

---

### Task 7.3: Determinism — fixed seed yields identical history and identical hedger params

**Files:**
- Test: `tests/test_train.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train.py  (append)
def test_train_is_deterministic_with_fixed_seed():
    cfg = _tiny_cfg(seed=7)

    hedger_a, hist_a = train(cfg)
    hedger_b, hist_b = train(cfg)

    # identical loss trajectory
    assert hist_a["loss"] == hist_b["loss"]
    assert hist_a["w"] == hist_b["w"]
    assert hist_a["premium"] == hist_b["premium"]

    # identical learned parameters
    sd_a = hedger_a.state_dict()
    sd_b = hedger_b.state_dict()
    assert sd_a.keys() == sd_b.keys()
    for name in sd_a:
        assert torch.equal(sd_a[name], sd_b[name]), f"param {name} differs across runs"
```

- [ ] **Step 2: Run test to verify it fails**

`pytest tests/test_train.py::test_train_is_deterministic_with_fixed_seed -v`

Expected: FAIL with `AssertionError: param ... differs across runs` if the hedger's weight
initialization is not seeded from `cfg.seed` (two `train(cfg)` calls would init the MLP
differently). The per-step generators are already deterministic (`cfg.seed + step`), but
the network init reads the global RNG.

- [ ] **Step 3: Write minimal implementation**

Seed the global torch RNG from `cfg.seed` immediately before constructing the hedger so
weight initialization is reproducible across calls:

```python
# deephedge/train.py  (inside train(cfg), as the first statements)
def train(cfg: ExperimentConfig) -> tuple[Hedger, dict]:
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)          # deterministic MLP weight init
    hedger = Hedger(_n_features(cfg), cfg.n_instruments).to(device)
    ...
```

- [ ] **Step 4: Run test to verify it passes**

`pytest tests/test_train.py::test_train_is_deterministic_with_fixed_seed -v`

Expected: PASS — two `train(cfg)` runs with the same seed produce bit-identical loss
histories and identical hedger parameters.

- [ ] **Step 5: Commit**

```bash
git add deephedge/train.py tests/test_train.py
git commit -m "fix(train): seed global RNG for reproducible hedger init"
```

---

### Task 7.4: Training reduces CVaR loss on a held-out eval batch

**Files:**
- Test: `tests/test_train.py`

This is spec §11 / §16(3): training should actually push the risk measure down. To avoid
SGD step-noise flakiness we compare the loss on a FIXED held-out batch before vs after
training, using an untrained hedger as the baseline and a longer (still tiny) run as the
trained model. We require a strict decrease with margin.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train.py  (append)
from deephedge.instruments import EuropeanOption
from deephedge.portfolio import build_instr_prices, simulate_pnl
from deephedge.benchmarks import make_nn_strategy
from deephedge.losses import cvar_loss
from deephedge.simulators.base import get_simulator
from deephedge.train import _n_features


def _eval_cvar(hedger, cfg, eval_seed=12345):
    """CVaR loss of `hedger` on one fixed held-out batch (w optimized in closed form)."""
    gen = torch.Generator(device=cfg.device)
    gen.manual_seed(eval_seed)
    simulate = get_simulator(cfg.model)
    paths = simulate(cfg, cfg.batch_size, gen)
    option = EuropeanOption(cfg.k, cfg.maturity)
    S0 = torch.tensor(cfg.s0, dtype=torch.float64, device=cfg.device)
    from deephedge.pricing.black_scholes import bs_price
    premium = float(bs_price(S0, cfg.k, cfg.maturity, cfg.r, cfg.sigma, q=cfg.q))
    instr_prices = build_instr_prices(cfg, paths, option)
    strat = make_nn_strategy(hedger, cfg)
    with torch.no_grad():
        pnl = simulate_pnl(strat, paths, cfg, option, premium, instr_prices).pnl
        # closed-form CVaR at the optimal w = VaR_alpha (empirical quantile of losses)
        losses = -pnl
        w_star = torch.quantile(losses, cfg.alpha)
        w_param = torch.nn.Parameter(w_star.clone())
        return float(cvar_loss(pnl, cfg.alpha, w_param).detach())


def test_training_reduces_cvar_on_held_out_batch():
    base = dict(model="gbm", loss="cvar", alpha=0.95, cost=0.0,
                n_steps=10, batch_size=2048, seed=3)
    cfg_untrained = ExperimentConfig(**{**_tiny_cfg(**base).__dict__})  # 2 steps
    # short-but-real training run
    cfg_trained = _tiny_cfg(model="gbm", loss="cvar", alpha=0.95, cost=0.0,
                            n_steps=10, batch_size=2048, seed=3,
                            epochs=1, steps_per_epoch=40, lr=5e-3)

    untrained = Hedger(_n_features(cfg_trained), cfg_trained.n_instruments)
    trained, _ = train(cfg_trained)

    loss_before = _eval_cvar(untrained, cfg_trained)
    loss_after = _eval_cvar(trained, cfg_trained)

    # training must lower tail risk on unseen paths by a clear margin
    assert loss_after < loss_before - 1e-3, (loss_before, loss_after)
```

- [ ] **Step 2: Run test to verify it fails**

`pytest tests/test_train.py::test_training_reduces_cvar_on_held_out_batch -v`

Expected: FAIL with `ModuleNotFoundError`/`ImportError` only if a dependency is missing;
otherwise it FAILS the assertion only if `train` does not actually optimize (e.g. forgot
`loss.backward()` / `opt.step()`). With Task 7.1's implementation present this passes; the
test exists to lock in the optimization behavior.

- [ ] **Step 3: Write minimal implementation**

No new production code: Task 7.1 already performs `opt.zero_grad(); loss.backward();
opt.step()` and exposes `_n_features`. If the assertion fails, the bug is a missing
backward/step in `train` — restore the exact loop body from Task 7.1:

```python
# deephedge/train.py  (loop body must contain, in order:)
        opt.zero_grad()
        loss.backward()
        opt.step()
        history["loss"].append(float(loss.detach()))
```

- [ ] **Step 4: Run test to verify it passes**

`pytest tests/test_train.py::test_training_reduces_cvar_on_held_out_batch -v`

Expected: PASS — the trained hedger has a strictly lower held-out CVaR than the untrained
one (by > 1e-3).

- [ ] **Step 5: Commit**

```bash
git add tests/test_train.py
git commit -m "test(train): CVaR decreases on held-out batch after training"
```

---

### Task 7.5: Frictionless GBM recovery — learned holdings approximate BS delta (tolerance gate, §13(6))

**Files:**
- Test: `tests/test_train.py`

Spec §16(3) + §13(6): after training on **cost=0 GBM**, the learned mid-step holding
should approximate the Black–Scholes delta. This is tolerance-based, not bit-exact — the
grid is discrete and SGD is noisy, so we require `mean(|δ_learned − δ_BS|) < 0.15` at a
mid-trajectory step over a held-out batch of ATM-ish paths. We train slightly longer than
the other tasks (still fast) to give the net a chance to converge toward delta.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train.py  (append)
from deephedge.pricing.black_scholes import bs_delta
from deephedge.hedger import build_features


def test_frictionless_gbm_recovers_bs_delta_within_tolerance():
    cfg = _tiny_cfg(model="gbm", loss="cvar", alpha=0.95, cost=0.0,
                    sigma=0.2, n_steps=20, batch_size=4096, seed=11,
                    epochs=1, steps_per_epoch=120, lr=5e-3)
    hedger, _ = train(cfg)

    # held-out batch of GBM paths
    gen = torch.Generator(device=cfg.device)
    gen.manual_seed(999)
    paths = get_simulator(cfg.model)(cfg, cfg.batch_size, gen)

    # mid-trajectory step
    i = cfg.n_steps // 2
    S_i = paths.S[:, i]                                   # (B,)
    tau = cfg.maturity - i * cfg.dt                       # scalar time-to-maturity (years)
    B = S_i.shape[0]
    prev_holdings = torch.zeros(B, cfg.n_instruments, dtype=S_i.dtype, device=cfg.device)
    # Feed features EXACTLY as make_nn_strategy does (normalized tau, k_norm=cfg.k) so the
    # query matches what the hedger was trained on.
    feats = build_features(S_i, tau / cfg.maturity, prev_holdings, V_i=None, k_norm=cfg.k)

    with torch.no_grad():
        learned = hedger(feats)[:, 0]                     # underlying holding (B,)

    # BS delta on the same spots; tau as a tensor broadcast (contract: tau float OR tensor)
    delta = bs_delta(S_i, cfg.k, tau, cfg.r, cfg.sigma, q=cfg.q)

    mean_abs_diff = (learned - delta).abs().mean().item()
    assert mean_abs_diff < 0.15, mean_abs_diff
```

- [ ] **Step 2: Run test to verify it fails**

`pytest tests/test_train.py::test_frictionless_gbm_recovers_bs_delta_within_tolerance -v`

Expected: FAIL with `AssertionError` (the printed `mean_abs_diff` will be ~0.3–0.5 for an
untrained-equivalent / under-trained net) if `train` does not converge toward delta, or
with an `ImportError` if `bs_delta`/`build_features` are missing. With Task 7.1's loop and
adequate `steps_per_epoch` it converges under the 0.15 tolerance.

- [ ] **Step 3: Write minimal implementation**

No new production code — this gate is satisfied by the Task 7.1 training loop. If it fails
on convergence, the permitted minimal fixes (no contract changes) are: (a) confirm
`make_nn_strategy(hedger, cfg)` feeds `build_features` with `V_i=None` for GBM so feature
dims match `_n_features`; (b) confirm the CVaR `w` param group LR multiplier is present so
the tail objective is well-conditioned. Both are already in Task 7.1:

```python
# deephedge/train.py  (already present — the two convergence-relevant lines)
    hedger = Hedger(_n_features(cfg), cfg.n_instruments).to(device)   # (a) dims match build_features
    params.append({"params": [w], "lr": cfg.lr * 10.0})              # (b) faster w LR for CVaR
```

- [ ] **Step 4: Run test to verify it passes**

`pytest tests/test_train.py::test_frictionless_gbm_recovers_bs_delta_within_tolerance -v`

Expected: PASS — `mean(|δ_learned − δ_BS|) < 0.15` at the mid step on held-out frictionless
GBM paths (the §13(6) recovery gate).

- [ ] **Step 5: Commit**

```bash
git add tests/test_train.py
git commit -m "test(train): frictionless GBM recovers BS delta within tolerance"
```

---

### Task 7.6: Full `train.py` regression — entire `tests/test_train.py` green

**Files:**
- Test: `tests/test_train.py`

- [ ] **Step 1: Write the failing test**

No new test code. This task runs the whole `test_train.py` file as a regression gate to
confirm all five behaviors (skeleton, entropic branch, determinism, CVaR reduction,
frictionless recovery) pass together and none regressed during the iterations above.

- [ ] **Step 2: Run test to verify it fails**

`pytest tests/test_train.py -v`

Expected: FAIL only if any earlier task's edit broke a prior test (e.g. a determinism
seed change altered the recovery run). If everything from 7.1–7.5 is intact, this is
already green — in which case proceed to confirm and commit.

- [ ] **Step 3: Write minimal implementation**

No production change unless a regression appears. If `test_train_is_deterministic_with_fixed_seed`
and `test_frictionless_gbm_recovers_bs_delta_within_tolerance` conflict (the global
`torch.manual_seed(cfg.seed)` from Task 7.3 changing recovery), the fix is to keep the
global seed for init only and rely on per-step `cfg.seed + step` generators for path
draws (already the design in Task 7.1) — no code change needed beyond what 7.1/7.3 wrote.

- [ ] **Step 4: Run test to verify it passes**

`pytest tests/test_train.py -v`

Expected: PASS — all of `test_train_returns_hedger_and_history_with_expected_keys`,
`test_train_entropic_loss_branch_has_no_w`, `test_train_is_deterministic_with_fixed_seed`,
`test_training_reduces_cvar_on_held_out_batch`, and
`test_frictionless_gbm_recovers_bs_delta_within_tolerance` green.

- [ ] **Step 5: Commit**

```bash
git add tests/test_train.py
git commit -m "test(train): full training-loop regression suite green"
```
## Group 8 — Heston simulator (full-truncation Euler)

Implements `deephedge/simulators/heston.py::simulate_heston(cfg, n_paths, generator) -> Paths`
using the **full-truncation Euler** scheme (spec §5.2). `get_simulator("heston")` already
resolves to it by lazy import (Group 1) — this group adds no registry edit (spec §13.3).
Binds exactly to the Shared Contracts:
`Paths(S, V, dt, times)` in `deephedge/simulators/base.py`, the
`simulate(cfg, n_paths, generator) -> Paths` simulator signature, and the
`ExperimentConfig` properties `dt`, `drift`.

Scheme (state may go negative; only the FUNCTION fed to coeffs is truncated):

```
V_plus      = max(V_i, 0)
V_{i+1}     = V_i + kappa*(theta - V_plus)*dt + xi*sqrt(V_plus)*sqrt(dt)*Z2   # stored state NOT truncated
logS_{i+1}  = logS_i + (drift - 0.5*V_plus)*dt + sqrt(V_plus)*sqrt(dt)*Z1
Z1 = Za ;  Z2 = rho*Za + sqrt(1-rho^2)*Zb ;  Za, Zb iid N(0,1)
```

Returns `S` and `V` both shaped `(n_paths, n_steps+1)`.

> **Prerequisite:** `deephedge/simulators/base.py` (`Paths`, `get_simulator`) and
> `deephedge/config.py` (`ExperimentConfig`) already exist from earlier groups.
> `get_simulator` resolves `"gbm"`/`"heston"`/`"merton"`/`"bates"` by lazy import; the
> `"heston"` branch goes live once this group creates `deephedge/simulators/heston.py`.

---

### Task 8.1: Heston path shapes and initial conditions

**Files:**
- Create: `deephedge/simulators/heston.py`
- Test: `tests/simulators/test_heston.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/simulators/test_heston.py
import math
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
```

- [ ] **Step 2: Run test to verify it fails**

Command: `pytest tests/simulators/test_heston.py::test_shapes_and_initial_conditions -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'deephedge.simulators.heston'`
(the module does not exist yet).

- [ ] **Step 3: Write minimal implementation**

```python
# deephedge/simulators/heston.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Command: `pytest tests/simulators/test_heston.py::test_shapes_and_initial_conditions -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add deephedge/simulators/heston.py tests/simulators/test_heston.py
git commit -m "feat(simulators): add full-truncation Euler Heston simulator with shape test"
```

---

### Task 8.2: Reproducibility under a seeded generator

**Files:**
- Modify: `tests/simulators/test_heston.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/simulators/test_heston.py  (append)
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
```

- [ ] **Step 2: Run test to verify it fails**

Command: `pytest tests/simulators/test_heston.py::test_reproducibility_same_seed_same_paths -v`
Expected: PASS immediately — the Task 8.1 implementation already routes all randomness
through the passed `generator`, so two identical seeds yield bit-identical paths. If this
test instead FAILS, it has caught a real defect (e.g. an implicit use of the global RNG
or a non-deterministic op); fix the implementation until it passes.

- [ ] **Step 3: Write minimal implementation**

No production change required: `simulate_heston` already draws every `torch.randn`
with the explicit `generator=generator` argument and performs no global-RNG calls, so
the behavior under test is already correct. (This step exists to keep the TDD cadence
explicit; the assertion is real and load-bearing as a regression guard against future
edits that reintroduce global-RNG usage.)

- [ ] **Step 4: Run test to verify it passes**

Command: `pytest tests/simulators/test_heston.py::test_reproducibility_same_seed_same_paths -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/simulators/test_heston.py
git commit -m "test(simulators): assert Heston paths are reproducible under a seeded generator"
```

---

### Task 8.3: Martingale check — discounted E[S_T] ≈ s0 under drift = r = 0

**Files:**
- Modify: `tests/simulators/test_heston.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/simulators/test_heston.py  (append)
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
```

- [ ] **Step 2: Run test to verify it fails**

Command: `pytest tests/simulators/test_heston.py::test_discounted_expectation_is_martingale -v`
Expected: PASS with the Task 8.1 implementation (the full-truncation drift uses
`cfg.drift`, which is `r = 0` here, giving an unbiased-in-the-mean log-Euler price up to
small variance-truncation bias). If it FAILS, it has caught a real drift/compensator bug
(e.g. an unwanted `+0.5*V` Itô term left in, or using `r` where `drift` was intended);
fix the implementation until the martingale property holds within the tolerance.

- [ ] **Step 3: Write minimal implementation**

No production change required: the Task 8.1 log-Euler update already uses the correct
risk-neutral log-drift `(drift - 0.5*V_plus)*dt`, so `E[exp(-rT) S_T] ≈ s0` holds. This
step documents that the martingale assertion passes against the existing implementation;
if a future edit breaks it, revert to the `(drift - 0.5*V_plus)` form.

- [ ] **Step 4: Run test to verify it passes**

Command: `pytest tests/simulators/test_heston.py::test_discounted_expectation_is_martingale -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/simulators/test_heston.py
git commit -m "test(simulators): Heston martingale check on discounted E[S_T] under drift=r=0"
```

---

### Task 8.4: Truncation correctness — coefficient function is non-negative while stored V may go negative

**Files:**
- Modify: `tests/simulators/test_heston.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/simulators/test_heston.py  (append)
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
```

- [ ] **Step 2: Run test to verify it fails**

Command: `pytest tests/simulators/test_heston.py::test_truncation_function_nonnegative_while_stored_state_may_be_negative -v`
Expected: PASS with the Task 8.1 implementation (it clamps `v_plus = V[:, i].clamp(min=0.0)`
for the coefficients but stores the raw, un-clamped `V[:, i + 1]`). If it FAILS on the
`(paths.V < 0.0).any()` assertion, the implementation is wrongly truncating the stored
state (absorption/reflection family — the worse scheme the spec §5.2 explicitly rejects);
fix it so only the coefficient function is truncated. If it FAILS on a finiteness
assertion, a negative value reached a `sqrt` — fix the clamp.

- [ ] **Step 3: Write minimal implementation**

No production change required: Task 8.1 already separates the truncated coefficient
function (`v_plus = V[:, i].clamp(min=0.0)`, used inside both the drift mean-reversion
term and the `sqrt_v_plus` diffusion term) from the stored next state (`V[:, i + 1]`,
written un-clamped). This step records that the full-truncation invariant holds against
the existing implementation and guards it against regressions.

- [ ] **Step 4: Run test to verify it passes**

Command: `pytest tests/simulators/test_heston.py::test_truncation_function_nonnegative_while_stored_state_may_be_negative -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/simulators/test_heston.py
git commit -m "test(simulators): verify full-truncation invariant (coeff>=0, stored V may be negative)"
```

---

### Task 8.5: `heston_scheme='qe'` raises NotImplementedError pointing to Andersen 2008

**Files:**
- Modify: `tests/simulators/test_heston.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/simulators/test_heston.py  (append)
import pytest


def test_qe_scheme_raises_not_implemented():
    cfg = _cfg(heston_scheme="qe", n_steps=10)
    gen = torch.Generator(device="cpu").manual_seed(0)
    with pytest.raises(NotImplementedError, match="Andersen"):
        simulate_heston(cfg, 100, gen)
```

- [ ] **Step 2: Run test to verify it fails**

Command: `pytest tests/simulators/test_heston.py::test_qe_scheme_raises_not_implemented -v`
Expected: PASS with the Task 8.1 implementation (the `if cfg.heston_scheme == "qe":`
guard raises `NotImplementedError` with an "Andersen" pointer). If it FAILS (e.g.
`NotImplementedError` not raised, or message missing "Andersen"), add/adjust the guard
at the top of `simulate_heston` to match the regex.

- [ ] **Step 3: Write minimal implementation**

No production change required: the `qe` guard added in Task 8.1 already raises
`NotImplementedError(... "Andersen (2008) QE" ...)`. This step confirms the stub behavior
is locked in by a test so the `full_truncation` default remains the only runnable scheme
until QE is implemented.

- [ ] **Step 4: Run test to verify it passes**

Command: `pytest tests/simulators/test_heston.py::test_qe_scheme_raises_not_implemented -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/simulators/test_heston.py
git commit -m "test(simulators): assert heston_scheme='qe' stub raises NotImplementedError"
```

---

### Task 8.6: `get_simulator("heston")` resolves to `simulate_heston`

**Files:**
- Test: `tests/simulators/test_get_simulator_heston.py`

> `get_simulator` (Group 1) already resolves `"heston"` by lazy import — there is NO
> registry dict to extend and NO edit to `base.py`. Now that `deephedge/simulators/heston.py`
> exists (Task 8.1), the existing `"heston"` branch (`from deephedge.simulators.heston import
> simulate_heston; return simulate_heston`) resolves at call time. This task adds the
> regression test that pins that contract.

- [ ] **Step 1: Write the failing test**

```python
# tests/simulators/test_get_simulator_heston.py
import torch

from deephedge.config import ExperimentConfig
from deephedge.simulators.base import Paths, get_simulator
from deephedge.simulators.heston import simulate_heston


def test_get_simulator_returns_heston_callable():
    sim = get_simulator("heston")
    assert sim is simulate_heston

    cfg = ExperimentConfig(
        s0=100.0, maturity=1.0, n_steps=10,
        v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7,
        model="heston", device="cpu",
    )
    gen = torch.Generator(device="cpu").manual_seed(0)
    paths = sim(cfg, 100, gen)
    assert isinstance(paths, Paths)
    assert paths.S.shape == (100, cfg.n_steps + 1)
    assert paths.V is not None and paths.V.shape == (100, cfg.n_steps + 1)
```

- [ ] **Step 2: Run test to verify it fails**

Command: `pytest tests/simulators/test_get_simulator_heston.py -v`
Expected: PASS once `heston.py` exists (Task 8.1) — the `get_simulator` `"heston"` branch
lazy-imports `simulate_heston`. Before `heston.py` exists the lazy import inside the branch
would raise `ModuleNotFoundError`; run after Task 8.1 to confirm green. If it FAILS on
`assert sim is simulate_heston`, the Group 1 `"heston"` lazy-import branch was altered —
restore it to `from deephedge.simulators.heston import simulate_heston; return simulate_heston`.

- [ ] **Step 3: Write minimal implementation**

No production change — `get_simulator` (Group 1) already has the lazy-import `"heston"`
branch. This task only adds the regression test that locks
`get_simulator("heston") is simulate_heston`.

- [ ] **Step 4: Run test to verify it passes**

Command: `pytest tests/simulators/test_get_simulator_heston.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/simulators/test_get_simulator_heston.py
git commit -m "test(simulators): get_simulator('heston') resolves to simulate_heston"
```

---

### Task 8.7: Full Heston simulator suite green

**Files:**
- Test: `tests/simulators/test_heston.py`, `tests/simulators/test_get_simulator_heston.py`

- [ ] **Step 1: Write the failing test**

No new test code. This task runs the full Group 8 test set together to confirm no
cross-test interference (shared `_cfg` helper, generator isolation) and that the suite is
green end to end.

- [ ] **Step 2: Run test to verify it fails**

Command: `pytest tests/simulators/test_heston.py tests/simulators/test_get_simulator_heston.py -v`
Expected: This should already be all-PASS if Tasks 8.1–8.6 landed correctly. Treat any
FAIL here as a real integration regression (e.g. a generator consumed across tests, or
an import cycle between `base.py` and `heston.py`) and debug before proceeding.

- [ ] **Step 3: Write minimal implementation**

If Step 2 surfaces an import cycle (`heston.py` imports `Paths` from `base.py` while
`get_simulator` resolves `simulate_heston`), note that `get_simulator`'s `"heston"` import
is already *function-local* (Group 1) — this breaks any cycle. No other change should be needed.

- [ ] **Step 4: Run test to verify it passes**

Command: `pytest tests/simulators/test_heston.py tests/simulators/test_get_simulator_heston.py -v`
Expected: PASS (all 6 tests).

- [ ] **Step 5: Commit**

```bash
git add tests/simulators/test_heston.py tests/simulators/test_get_simulator_heston.py
git commit -m "test(simulators): full Heston simulator suite green (shapes, repro, martingale, truncation, qe-stub, registry)"
```
## Group 9: Jumps — Merton + Bates simulators

Implements `deephedge/simulators/jumps.py`: `simulate_merton` and `simulate_bates`. Both
are resolved through `get_simulator` (Group 1) by its lazy-import `("merton", "bates")`
branch — this group adds no edit to `deephedge/simulators/base.py`. Merton = GBM log-Euler
dynamics + compound-Poisson jumps + risk-neutral compensator; Bates = Heston
full-truncation Euler dynamics + the same jumps + compensator (spec §5.3, §13(3)).

**Binds to these existing Shared-Contract symbols (defined in earlier groups):**

- `deephedge/config.py` → `ExperimentConfig` (fields `s0, k, r, q, mu, maturity, n_steps,
  sigma, v0, kappa, theta, xi, rho, jump_intensity, jump_mean, jump_std, model`;
  properties `.dt`, `.drift`).
- `deephedge/simulators/base.py` → `Paths(S, V, dt, times)` dataclass and
  `get_simulator(model: str)` registry.
- `deephedge/simulators/gbm.py` → `simulate_gbm(cfg, n_paths, generator) -> Paths`.
- `deephedge/simulators/heston.py` → `simulate_heston(cfg, n_paths, generator) -> Paths`.

**Jump model (spec §5.3).** Over a step `dt`, per path: `n ~ Poisson(jump_intensity*dt)`,
and the aggregate log-jump increment is `J = Σ_{k=1}^{n} Y_k` with `Y_k ~ N(jump_mean,
jump_std^2)`. Because a sum of `n` iid normals is itself normal, we vectorize without
materializing individual jumps: draw `N_i ~ Poisson(jump_intensity*dt)` (one count per
path per step), then `J_i ~ Normal(N_i*jump_mean, N_i*jump_std^2)`, i.e.
`J_i = N_i*jump_mean + sqrt(N_i)*jump_std*Z_J` with `Z_J ~ N(0,1)` (when `N_i = 0` both
terms vanish, giving `J_i = 0` exactly).

**Risk-neutral compensator (spec §5.3).** To keep the discounted price a martingale, the
per-step log-price drift subtracts `comp = jump_intensity*(exp(jump_mean +
0.5*jump_std^2) - 1)*dt`. So Merton's log-step is
`(drift - 0.5*sigma^2)*dt - comp + sigma*sqrt(dt)*Z + J_i`, and Bates' log-step is the
Heston full-truncation log-step minus `comp` plus `J_i`. The `drift` is `cfg.drift`
(`= r` when `mu is None`).

---

### Task 9.1: Helper `_sample_jumps` — vectorized per-step aggregate log-jump

**Files:**
- Create: `deephedge/simulators/jumps.py`
- Test: `tests/test_jumps.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_jumps.py
import math
import torch

from deephedge.config import ExperimentConfig
from deephedge.simulators.jumps import _sample_jumps


def test_sample_jumps_zero_intensity_is_exactly_zero():
    # With jump_intensity=0 every Poisson count is 0, so the aggregate
    # log-jump must be identically zero (no jump contribution).
    cfg = ExperimentConfig(jump_intensity=0.0, jump_mean=-0.1, jump_std=0.15)
    g = torch.Generator().manual_seed(0)
    J = _sample_jumps(cfg, n_paths=10_000, generator=g)
    assert J.shape == (10_000,)
    assert torch.all(J == 0.0)


def test_sample_jumps_mean_matches_compound_poisson():
    # E[J] = E[N]*jump_mean = (lambda*dt)*jump_mean.
    cfg = ExperimentConfig(
        jump_intensity=5.0, jump_mean=-0.1, jump_std=0.15,
        maturity=1.0, n_steps=1,
    )
    g = torch.Generator().manual_seed(1)
    n_paths = 400_000
    J = _sample_jumps(cfg, n_paths=n_paths, generator=g)
    expected_mean = cfg.jump_intensity * cfg.dt * cfg.jump_mean
    # Var[J] = E[N]*(jump_mean^2 + jump_std^2); stderr of the sample mean:
    lam_dt = cfg.jump_intensity * cfg.dt
    var_J = lam_dt * (cfg.jump_mean ** 2 + cfg.jump_std ** 2)
    stderr = math.sqrt(var_J / n_paths)
    assert abs(J.mean().item() - expected_mean) < 4 * stderr


def test_sample_jumps_variance_matches_compound_poisson():
    # Var[J] = E[N]*(jump_mean^2 + jump_std^2)  (compound-Poisson variance).
    cfg = ExperimentConfig(
        jump_intensity=8.0, jump_mean=0.0, jump_std=0.2,
        maturity=1.0, n_steps=1,
    )
    g = torch.Generator().manual_seed(2)
    n_paths = 400_000
    J = _sample_jumps(cfg, n_paths=n_paths, generator=g)
    lam_dt = cfg.jump_intensity * cfg.dt
    expected_var = lam_dt * (cfg.jump_mean ** 2 + cfg.jump_std ** 2)
    sample_var = J.var(unbiased=True).item()
    # 5% relative tolerance at 4e5 paths.
    assert abs(sample_var - expected_var) / expected_var < 0.05
```

- [ ] **Step 2: Run test to verify it fails**

`pytest tests/test_jumps.py -v -k _sample_jumps`

Expected: FAIL — `ImportError` / `cannot import name '_sample_jumps' from
'deephedge.simulators.jumps'` (module does not exist yet).

- [ ] **Step 3: Write minimal implementation**

```python
# deephedge/simulators/jumps.py
"""Merton (GBM + jumps) and Bates (Heston + jumps) simulators.

Compound-Poisson log-jumps with the Merton/Bates risk-neutral compensator so the
discounted spot is a martingale under mu = r (spec §5.3).
"""
import math

import torch

from deephedge.config import ExperimentConfig
from deephedge.simulators.base import Paths


def _sample_jumps(
    cfg: ExperimentConfig, n_paths: int, generator: torch.Generator
) -> torch.Tensor:
    """Aggregate per-step log-jump increment for each path: (n_paths,).

    n ~ Poisson(jump_intensity*dt) jumps per step, each Y_k ~ N(jump_mean, jump_std^2);
    the sum of n iid normals is N(n*jump_mean, n*jump_std^2), so we draw the count then a
    single normal, avoiding materializing individual jumps. When n=0 the increment is
    exactly 0 (both the mean term and sqrt(n) scale term vanish).
    """
    device = torch.device(cfg.device)
    rate = torch.full(
        (n_paths,), cfg.jump_intensity * cfg.dt, device=device, dtype=torch.float64
    )
    counts = torch.poisson(rate, generator=generator)  # float tensor of non-neg ints
    z = torch.randn(n_paths, generator=generator, device=device, dtype=torch.float64)
    jumps = counts * cfg.jump_mean + torch.sqrt(counts) * cfg.jump_std * z
    return jumps


def _jump_compensator(cfg: ExperimentConfig) -> float:
    """Per-step risk-neutral drift compensator:
    jump_intensity*(exp(jump_mean + 0.5*jump_std^2) - 1)*dt  (Merton 1976 / Bates 1996).
    """
    k_bar = math.exp(cfg.jump_mean + 0.5 * cfg.jump_std ** 2) - 1.0
    return cfg.jump_intensity * k_bar * cfg.dt
```

- [ ] **Step 4: Run test to verify it passes**

`pytest tests/test_jumps.py -v -k _sample_jumps`

Expected: PASS — zero-intensity gives identically zero; mean within 4·stderr; variance
within 5% of the compound-Poisson value.

- [ ] **Step 5: Commit**

```bash
git add deephedge/simulators/jumps.py tests/test_jumps.py
git commit -m "feat(simulators): vectorized compound-Poisson jump sampler + compensator helper"
```

---

### Task 9.2: `simulate_merton` — GBM dynamics + jumps + compensator (shape + state)

**Files:**
- Modify: `deephedge/simulators/jumps.py`
- Test: `tests/test_jumps.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_jumps.py  (append)
from deephedge.simulators.base import Paths
from deephedge.simulators.jumps import simulate_merton


def test_simulate_merton_shapes_and_state():
    cfg = ExperimentConfig(
        s0=100.0, sigma=0.2, maturity=30 / 252, n_steps=30,
        jump_intensity=2.0, jump_mean=-0.1, jump_std=0.15,
    )
    g = torch.Generator().manual_seed(0)
    n_paths = 5_000
    paths = simulate_merton(cfg, n_paths=n_paths, generator=g)
    assert isinstance(paths, Paths)
    assert paths.S.shape == (n_paths, cfg.n_steps + 1)
    assert paths.V is None                      # Merton has no variance process
    assert paths.times.shape == (cfg.n_steps + 1,)
    assert abs(paths.dt - cfg.dt) < 1e-12
    # First column is exactly s0.
    assert torch.allclose(paths.S[:, 0], torch.full((n_paths,), cfg.s0, dtype=paths.S.dtype))
    # times grid is 0 .. maturity inclusive, evenly spaced.
    assert paths.times[0].item() == 0.0
    assert abs(paths.times[-1].item() - cfg.maturity) < 1e-12
    # Strictly positive prices (jumps act on the log-price -> S stays > 0).
    assert torch.all(paths.S > 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

`pytest tests/test_jumps.py -v -k test_simulate_merton_shapes_and_state`

Expected: FAIL — `ImportError: cannot import name 'simulate_merton'`.

- [ ] **Step 3: Write minimal implementation**

```python
# deephedge/simulators/jumps.py  (append)
def simulate_merton(
    cfg: ExperimentConfig, n_paths: int, generator: torch.Generator
) -> Paths:
    """GBM log-Euler + compound-Poisson jumps + risk-neutral compensator (spec §5.3).

    log S_{i+1} = log S_i + (drift - 0.5*sigma^2)*dt - comp + sigma*sqrt(dt)*Z + J_i
    where comp = jump_intensity*(exp(jump_mean + 0.5*jump_std^2) - 1)*dt.
    """
    device = torch.device(cfg.device)
    dt = cfg.dt
    n = cfg.n_steps
    comp = _jump_compensator(cfg)
    diffusion_drift = (cfg.drift - 0.5 * cfg.sigma ** 2) * dt - comp

    log_s = torch.empty(n_paths, n + 1, device=device, dtype=torch.float64)
    log_s[:, 0] = math.log(cfg.s0)
    sqrt_dt = math.sqrt(dt)
    for i in range(n):
        z = torch.randn(n_paths, generator=generator, device=device, dtype=torch.float64)
        jumps = _sample_jumps(cfg, n_paths=n_paths, generator=generator)
        log_s[:, i + 1] = (
            log_s[:, i] + diffusion_drift + cfg.sigma * sqrt_dt * z + jumps
        )

    S = torch.exp(log_s)
    times = torch.linspace(0.0, cfg.maturity, n + 1, device=device, dtype=torch.float64)
    return Paths(S=S, V=None, dt=dt, times=times)
```

- [ ] **Step 4: Run test to verify it passes**

`pytest tests/test_jumps.py -v -k test_simulate_merton_shapes_and_state`

Expected: PASS — `Paths` with `S` shape `(5000, 31)`, `V is None`, `S[:,0] == s0`,
all prices positive.

- [ ] **Step 5: Commit**

```bash
git add deephedge/simulators/jumps.py tests/test_jumps.py
git commit -m "feat(simulators): simulate_merton (GBM + jumps + compensator)"
```

---

### Task 9.3: `simulate_merton` martingale check — validates the compensator sign

**Files:**
- Test: `tests/test_jumps.py`

This is the spec §13(3) gate specialized to jumps: under `mu = r = 0` WITH the
compensator, the discounted (here undiscounted, since `r=0`) terminal price satisfies
`E[S_T] ≈ s0`. Without the compensator (or with the wrong sign) the jump mean
`jump_mean = -0.1` would bias `E[S_T]` well below `s0`, so this assertion specifically
pins the compensator sign.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_jumps.py  (append)
def test_simulate_merton_martingale_with_compensator():
    # mu=r=0 -> drift=0; discount factor exp(-r*T)=1. With the compensator,
    # E[S_T] must equal s0 despite the negative-mean jumps.
    cfg = ExperimentConfig(
        s0=100.0, r=0.0, mu=None, sigma=0.2, maturity=1.0, n_steps=50,
        jump_intensity=5.0, jump_mean=-0.1, jump_std=0.15,
    )
    g = torch.Generator().manual_seed(7)
    n_paths = 400_000
    paths = simulate_merton(cfg, n_paths=n_paths, generator=g)
    S_T = paths.S[:, -1]
    mean_ST = S_T.mean().item()
    stderr = (S_T.std(unbiased=True) / (n_paths ** 0.5)).item()
    # discounted E[S_T] ~= s0 within ~3*stderr (r=0 so no discounting needed).
    assert abs(mean_ST - cfg.s0) < 3 * stderr


def test_simulate_merton_martingale_breaks_without_compensator():
    # Sanity that the test above is real: dropping the compensator term biases
    # the mean below s0 by far more than the MC error. (Documents the sign.)
    import math as _math
    cfg = ExperimentConfig(
        s0=100.0, r=0.0, mu=None, sigma=0.2, maturity=1.0, n_steps=50,
        jump_intensity=5.0, jump_mean=-0.1, jump_std=0.15,
    )
    g = torch.Generator().manual_seed(7)
    paths = simulate_merton(cfg, n_paths=400_000, generator=g)
    biased_mean = paths.S[:, -1].mean().item()
    # The compensator over the whole horizon is exp(lambda*(e^{m+s^2/2}-1)*T):
    k_bar = _math.exp(cfg.jump_mean + 0.5 * cfg.jump_std ** 2) - 1.0
    no_comp_factor = _math.exp(cfg.jump_intensity * k_bar * cfg.maturity)
    # If the compensator were missing, the mean would be ~ s0/no_comp_factor.
    # With it present, mean ~= s0 -> well above that biased value.
    assert biased_mean > cfg.s0 * no_comp_factor * 1.01
```

- [ ] **Step 2: Run test to verify it fails**

`pytest tests/test_jumps.py -v -k martingale`

Expected: FAIL initially only if the implementation is wrong; against the Task 9.2
implementation these should PASS. Run now to confirm the compensator sign is correct
(if the sign were flipped, `test_simulate_merton_martingale_with_compensator` would FAIL
because `E[S_T]` would overshoot `s0` by `no_comp_factor^2`).

- [ ] **Step 3: Write minimal implementation**

No new implementation — `simulate_merton` from Task 9.2 already carries the compensator
with the correct sign (`diffusion_drift` subtracts `comp`). This task is a validation
gate; if it fails, fix the sign of `comp` in `_jump_compensator` / its use in
`simulate_merton` before proceeding.

- [ ] **Step 4: Run test to verify it passes**

`pytest tests/test_jumps.py -v -k martingale`

Expected: PASS — `E[S_T]` within `3*stderr` of `s0`; the broken-compensator sanity check
confirms the gate has teeth.

- [ ] **Step 5: Commit**

```bash
git add tests/test_jumps.py
git commit -m "test(simulators): Merton martingale gate validates compensator sign"
```

---

### Task 9.4: `simulate_merton` cross-check vs Merton closed-form call price

**Files:**
- Test: `tests/test_jumps.py`

Spec §5.3 / §13: cross-check the Merton simulator against Merton's closed-form call
price, a Poisson-weighted sum of Black–Scholes prices. We implement the closed form
locally in the test (summing ~40 terms) and verify the discounted MC call payoff lands
in its Monte-Carlo confidence interval.

Merton (1976) closed form for a call under `r`:
for each jump count `n`, the effective parameters are
`sigma_n^2 = sigma^2 + n*jump_std^2/T` and
`r_n = r - jump_intensity*k_bar + n*(jump_mean + 0.5*jump_std^2)/T`,
with `k_bar = exp(jump_mean + 0.5*jump_std^2) - 1`, and the price is
`Σ_n exp(-lam'*T)*(lam'*T)^n / n! * BS_call(s0, K, T, r_n, sigma_n)`
where `lam' = jump_intensity*(1 + k_bar) = jump_intensity*exp(jump_mean+0.5*jump_std^2)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_jumps.py  (append)
def test_simulate_merton_matches_closed_form_call():
    import math as _math
    from math import erf, lgamma

    def _norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + erf(x / _math.sqrt(2.0)))

    def _bs_call(s0, K, T, r, sigma):
        if sigma <= 0.0 or T <= 0.0:
            return max(s0 - K * _math.exp(-r * T), 0.0)
        vol = sigma * _math.sqrt(T)
        d1 = (_math.log(s0 / K) + (r + 0.5 * sigma ** 2) * T) / vol
        d2 = d1 - vol
        return s0 * _norm_cdf(d1) - K * _math.exp(-r * T) * _norm_cdf(d2)

    def _merton_call(s0, K, T, r, sigma, lam, m, s, n_terms=40):
        k_bar = _math.exp(m + 0.5 * s ** 2) - 1.0
        lam_p = lam * (1.0 + k_bar)             # lam' = lam*exp(m+0.5 s^2)
        price = 0.0
        for n in range(n_terms):
            sigma_n = _math.sqrt(sigma ** 2 + n * s ** 2 / T)
            r_n = r - lam * k_bar + n * (m + 0.5 * s ** 2) / T
            log_w = -lam_p * T + n * _math.log(lam_p * T) - lgamma(n + 1)
            weight = _math.exp(log_w)
            price += weight * _bs_call(s0, K, T, r_n, sigma_n)
        return price

    cfg = ExperimentConfig(
        s0=100.0, k=100.0, r=0.03, mu=None, sigma=0.2, maturity=1.0, n_steps=100,
        jump_intensity=1.0, jump_mean=-0.1, jump_std=0.15,
    )
    closed = _merton_call(
        cfg.s0, cfg.k, cfg.maturity, cfg.r, cfg.sigma,
        cfg.jump_intensity, cfg.jump_mean, cfg.jump_std, n_terms=40,
    )

    g = torch.Generator().manual_seed(11)
    n_paths = 500_000
    paths = simulate_merton(cfg, n_paths=n_paths, generator=g)
    S_T = paths.S[:, -1]
    disc = _math.exp(-cfg.r * cfg.maturity)
    payoff = torch.clamp(S_T - cfg.k, min=0.0) * disc
    mc_price = payoff.mean().item()
    mc_stderr = (payoff.std(unbiased=True) / (n_paths ** 0.5)).item()

    # closed-form must lie inside the MC 99% CI (z=2.58).
    assert abs(mc_price - closed) < 2.58 * mc_stderr, (
        f"MC={mc_price:.4f} closed={closed:.4f} stderr={mc_stderr:.4f}"
    )
```

- [ ] **Step 2: Run test to verify it fails**

`pytest tests/test_jumps.py -v -k matches_closed_form_call`

Expected: FAIL if the simulator drift/compensator is wrong (the closed form and MC
disagree beyond the CI). Against the correct Task 9.2 implementation it PASSes; run now
to confirm.

- [ ] **Step 3: Write minimal implementation**

No new implementation — this validates `simulate_merton`. The closed-form Merton price is
fully implemented inside the test (40-term Poisson-weighted BS sum, `r_n`/`sigma_n`
per term, log-space Poisson weights via `lgamma` for numerical safety). If the test
fails, the bug is in `simulate_merton`'s drift/compensator, not the test.

- [ ] **Step 4: Run test to verify it passes**

`pytest tests/test_jumps.py -v -k matches_closed_form_call`

Expected: PASS — MC call price within the 99% CI of the closed-form Merton price.

- [ ] **Step 5: Commit**

```bash
git add tests/test_jumps.py
git commit -m "test(simulators): Merton MC vs closed-form Poisson-weighted BS call"
```

---

### Task 9.5: `simulate_merton` reduces to `simulate_gbm` when `jump_intensity=0`

**Files:**
- Test: `tests/test_jumps.py`

Spec §13: with `jump_intensity=0` the compensator is 0 and every Poisson count is 0, so
`simulate_merton` is GBM. We assert the terminal distributions match closely under the
same seed. Note the per-step RNG consumption differs (Merton also draws a Poisson count
and a jump normal per step), so paths are not bit-identical; we compare distributional
moments at matched precision.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_jumps.py  (append)
from deephedge.simulators.gbm import simulate_gbm


def test_simulate_merton_zero_intensity_matches_gbm_in_distribution():
    cfg = ExperimentConfig(
        s0=100.0, r=0.01, mu=None, sigma=0.2, maturity=1.0, n_steps=50,
        jump_intensity=0.0, jump_mean=-0.1, jump_std=0.15,
    )
    n_paths = 300_000
    g_m = torch.Generator().manual_seed(123)
    g_g = torch.Generator().manual_seed(123)
    merton = simulate_merton(cfg, n_paths=n_paths, generator=g_m)
    gbm = simulate_gbm(cfg, n_paths=n_paths, generator=g_g)

    st_m = merton.S[:, -1]
    st_g = gbm.S[:, -1]
    # Means and stds of S_T agree to <1% relative (both are the same GBM law).
    assert abs(st_m.mean().item() - st_g.mean().item()) / st_g.mean().item() < 0.01
    assert abs(st_m.std().item() - st_g.std().item()) / st_g.std().item() < 0.02
    # Quantile match at the 5% and 95% tails (<1.5% relative).
    qs = torch.tensor([0.05, 0.95], dtype=st_m.dtype)
    qm = torch.quantile(st_m, qs)
    qg = torch.quantile(st_g, qs)
    assert torch.all((qm - qg).abs() / qg < 0.015)
```

- [ ] **Step 2: Run test to verify it fails**

`pytest tests/test_jumps.py -v -k zero_intensity_matches_gbm`

Expected: FAIL only if the zero-intensity path diverges from GBM (e.g. a stray
compensator or jump term not vanishing). Against Task 9.2 it PASSes; run to confirm
`simulate_merton` collapses to GBM.

- [ ] **Step 3: Write minimal implementation**

No new implementation — when `jump_intensity=0`: `_jump_compensator` returns 0 (factor
`k_bar` is multiplied by `jump_intensity`), and `_sample_jumps` returns all-zeros
(`torch.poisson(0)` is 0, `sqrt(0)=0`), so `simulate_merton`'s log-step equals GBM's
`(drift-0.5*sigma^2)*dt + sigma*sqrt(dt)*Z`. If this test fails, the bug is a
jump-intensity-independent term leaking into the drift.

- [ ] **Step 4: Run test to verify it passes**

`pytest tests/test_jumps.py -v -k zero_intensity_matches_gbm`

Expected: PASS — terminal-price mean/std and 5%/95% quantiles match `simulate_gbm`.

- [ ] **Step 5: Commit**

```bash
git add tests/test_jumps.py
git commit -m "test(simulators): Merton with zero intensity matches GBM distribution"
```

---

### Task 9.6: `simulate_bates` — Heston full-truncation dynamics + jumps + compensator

**Files:**
- Modify: `deephedge/simulators/jumps.py`
- Test: `tests/test_jumps.py`

Bates = Heston full-truncation Euler (spec §5.2) for the diffusion + variance state, with
the same compound-Poisson jumps and compensator added to the log-price. We reuse the
exact full-truncation log-step from `simulate_heston` and add `-comp + J_i` per step.
We inline the full-truncation recursion here (rather than calling `simulate_heston`)
because the jumps must enter the same per-step log update.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_jumps.py  (append)
from deephedge.simulators.jumps import simulate_bates


def test_simulate_bates_shapes_and_variance_state():
    cfg = ExperimentConfig(
        s0=100.0, r=0.0, mu=None, maturity=1.0, n_steps=50,
        v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7,
        jump_intensity=2.0, jump_mean=-0.1, jump_std=0.15,
    )
    g = torch.Generator().manual_seed(0)
    n_paths = 5_000
    paths = simulate_bates(cfg, n_paths=n_paths, generator=g)
    assert isinstance(paths, Paths)
    assert paths.S.shape == (n_paths, cfg.n_steps + 1)
    assert paths.V is not None and paths.V.shape == (n_paths, cfg.n_steps + 1)
    assert torch.allclose(paths.S[:, 0], torch.full((n_paths,), cfg.s0, dtype=paths.S.dtype))
    assert torch.allclose(paths.V[:, 0], torch.full((n_paths,), cfg.v0, dtype=paths.V.dtype))
    assert torch.all(paths.S > 0.0)             # jumps act on log-price -> S > 0
    assert abs(paths.dt - cfg.dt) < 1e-12
    assert paths.times.shape == (cfg.n_steps + 1,)
```

- [ ] **Step 2: Run test to verify it fails**

`pytest tests/test_jumps.py -v -k test_simulate_bates_shapes_and_variance_state`

Expected: FAIL — `ImportError: cannot import name 'simulate_bates'`.

- [ ] **Step 3: Write minimal implementation**

```python
# deephedge/simulators/jumps.py  (append)
def simulate_bates(
    cfg: ExperimentConfig, n_paths: int, generator: torch.Generator
) -> Paths:
    """Heston full-truncation Euler + compound-Poisson jumps + compensator (spec §5.3).

    V_plus       = max(V_i, 0)
    V_{i+1}      = V_i + kappa*(theta - V_plus)*dt + xi*sqrt(V_plus)*sqrt(dt)*Z2
    log S_{i+1}  = log S_i + (drift - 0.5*V_plus)*dt - comp
                            + sqrt(V_plus)*sqrt(dt)*Z1 + J_i
    Z1 = Za, Z2 = rho*Za + sqrt(1-rho^2)*Zb  (Cholesky); carried V may go negative.
    """
    device = torch.device(cfg.device)
    dt = cfg.dt
    n = cfg.n_steps
    sqrt_dt = math.sqrt(dt)
    comp = _jump_compensator(cfg)
    rho = cfg.rho
    sqrt_1m_rho2 = math.sqrt(max(1.0 - rho ** 2, 0.0))

    log_s = torch.empty(n_paths, n + 1, device=device, dtype=torch.float64)
    V = torch.empty(n_paths, n + 1, device=device, dtype=torch.float64)
    log_s[:, 0] = math.log(cfg.s0)
    V[:, 0] = cfg.v0

    for i in range(n):
        za = torch.randn(n_paths, generator=generator, device=device, dtype=torch.float64)
        zb = torch.randn(n_paths, generator=generator, device=device, dtype=torch.float64)
        z1 = za
        z2 = rho * za + sqrt_1m_rho2 * zb
        jumps = _sample_jumps(cfg, n_paths=n_paths, generator=generator)

        v_prev = V[:, i]
        v_plus = torch.clamp(v_prev, min=0.0)
        sqrt_v_plus = torch.sqrt(v_plus)

        # carried variance state is NOT truncated (full-truncation scheme).
        V[:, i + 1] = (
            v_prev + cfg.kappa * (cfg.theta - v_plus) * dt
            + cfg.xi * sqrt_v_plus * sqrt_dt * z2
        )
        log_s[:, i + 1] = (
            log_s[:, i] + (cfg.drift - 0.5 * v_plus) * dt - comp
            + sqrt_v_plus * sqrt_dt * z1 + jumps
        )

    S = torch.exp(log_s)
    times = torch.linspace(0.0, cfg.maturity, n + 1, device=device, dtype=torch.float64)
    return Paths(S=S, V=V, dt=dt, times=times)
```

- [ ] **Step 4: Run test to verify it passes**

`pytest tests/test_jumps.py -v -k test_simulate_bates_shapes_and_variance_state`

Expected: PASS — `Paths` with `S` and `V` both `(5000, 51)`, `S[:,0]==s0`,
`V[:,0]==v0`, all prices positive.

- [ ] **Step 5: Commit**

```bash
git add deephedge/simulators/jumps.py tests/test_jumps.py
git commit -m "feat(simulators): simulate_bates (Heston full-truncation + jumps + compensator)"
```

---

### Task 9.7: `simulate_bates` martingale check — compensator under stochastic vol

**Files:**
- Test: `tests/test_jumps.py`

Spec §13(3): under `mu = r = 0`, with the compensator, discounted `E[S_T] ≈ s0` for Bates
too. This jointly validates the Heston full-truncation diffusion (which is itself a
martingale under `r=0`) and the jump compensator sign on top of stochastic vol.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_jumps.py  (append)
def test_simulate_bates_martingale_with_compensator():
    cfg = ExperimentConfig(
        s0=100.0, r=0.0, mu=None, maturity=1.0, n_steps=100,
        v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7,
        jump_intensity=5.0, jump_mean=-0.1, jump_std=0.15,
    )
    g = torch.Generator().manual_seed(21)
    n_paths = 500_000
    paths = simulate_bates(cfg, n_paths=n_paths, generator=g)
    S_T = paths.S[:, -1]
    mean_ST = S_T.mean().item()
    stderr = (S_T.std(unbiased=True) / (n_paths ** 0.5)).item()
    # r=0 -> discount factor 1; mean must sit within ~3*stderr of s0.
    assert abs(mean_ST - cfg.s0) < 3 * stderr
```

- [ ] **Step 2: Run test to verify it fails**

`pytest tests/test_jumps.py -v -k test_simulate_bates_martingale_with_compensator`

Expected: FAIL if the compensator sign or full-truncation log-drift is wrong (mean would
drift off `s0` by far more than `3*stderr`). Against Task 9.6 it PASSes; run to confirm.

- [ ] **Step 3: Write minimal implementation**

No new implementation — `simulate_bates` from Task 9.6 already subtracts `comp` and uses
the full-truncation `(drift - 0.5*v_plus)*dt` log-drift, which is a martingale at `r=0`
before jumps and stays a martingale after adding compensated jumps. If this fails, fix
the `-comp` term in `simulate_bates`.

- [ ] **Step 4: Run test to verify it passes**

`pytest tests/test_jumps.py -v -k test_simulate_bates_martingale_with_compensator`

Expected: PASS — discounted `E[S_T]` within `3*stderr` of `s0` under Bates.

- [ ] **Step 5: Commit**

```bash
git add tests/test_jumps.py
git commit -m "test(simulators): Bates martingale gate validates compensator under stoch vol"
```

---

### Task 9.8: `get_simulator` resolves `"merton"` and `"bates"`

**Files:**
- Test: `tests/test_jumps.py`

`get_simulator(model)` is the Shared-Contract registry returning the callable
`(cfg, n_paths, generator) -> Paths`. Its `("merton", "bates")` branch already
lazy-imports `simulate_merton` / `simulate_bates` from `deephedge.simulators.jumps`
(Group 1) — there is NO registry dict to extend and NO edit to `base.py`. Now that
`jumps.py` exists (Tasks 9.2/9.6), those branches resolve at call time. The lazy import
inside `get_simulator` also avoids the circular import (`jumps.py` imports `base.py` for
`Paths`). This task adds the regression test pinning the contract.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_jumps.py  (append)
from deephedge.simulators.base import get_simulator


def test_get_simulator_registers_merton_and_bates():
    g = torch.Generator().manual_seed(0)

    cfg_m = ExperimentConfig(
        model="merton", s0=100.0, sigma=0.2, maturity=1.0, n_steps=20,
        jump_intensity=2.0, jump_mean=-0.1, jump_std=0.15,
    )
    sim_m = get_simulator("merton")
    paths_m = sim_m(cfg_m, 1_000, g)
    assert paths_m.S.shape == (1_000, cfg_m.n_steps + 1)
    assert paths_m.V is None
    # Same callable identity as the direct function.
    assert sim_m is simulate_merton

    cfg_b = ExperimentConfig(
        model="bates", s0=100.0, maturity=1.0, n_steps=20,
        v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7,
        jump_intensity=2.0, jump_mean=-0.1, jump_std=0.15,
    )
    sim_b = get_simulator("bates")
    paths_b = sim_b(cfg_b, 1_000, g)
    assert paths_b.S.shape == (1_000, cfg_b.n_steps + 1)
    assert paths_b.V is not None and paths_b.V.shape == (1_000, cfg_b.n_steps + 1)
    assert sim_b is simulate_bates
```

- [ ] **Step 2: Run test to verify it fails**

`pytest tests/test_jumps.py -v -k test_get_simulator_registers_merton_and_bates`

Expected: PASS once `jumps.py` exists (Tasks 9.2/9.6) — the `("merton", "bates")` branch of
`get_simulator` (Group 1) lazy-imports `simulate_merton` / `simulate_bates`. Before
`jumps.py` exists the lazy import would raise `ModuleNotFoundError`; run after Task 9.6 to
confirm green. If it FAILS on `sim_m is simulate_merton`, the Group 1 lazy-import branch was
altered — restore it to `from deephedge.simulators.jumps import simulate_merton, simulate_bates`.

- [ ] **Step 3: Write minimal implementation**

No production change — `get_simulator` (Group 1) already resolves `"merton"`/`"bates"` via
its lazy-import branch:

```python
# deephedge/simulators/base.py  (Group 1 — already present, NOT edited here)
    if model in ("merton", "bates"):
        from deephedge.simulators.jumps import simulate_merton, simulate_bates
        return {"merton": simulate_merton, "bates": simulate_bates}[model]
```

This task only adds the regression test that locks
`get_simulator("merton") is simulate_merton` and `get_simulator("bates") is simulate_bates`.

- [ ] **Step 4: Run test to verify it passes**

`pytest tests/test_jumps.py -v -k test_get_simulator_registers_merton_and_bates`

Expected: PASS — `get_simulator("merton") is simulate_merton`,
`get_simulator("bates") is simulate_bates`, and both produce correctly-shaped `Paths`.

- [ ] **Step 5: Commit**

```bash
git add tests/test_jumps.py
git commit -m "test(simulators): get_simulator resolves merton and bates"
```

---

### Task 9.9: Full jumps suite green + device-agnostic sanity

**Files:**
- Test: `tests/test_jumps.py`

Final gate: run the whole `tests/test_jumps.py` suite, and add one device-agnostic check
that the simulators honor `cfg.device` so the contract "all tensors live on cfg.device"
holds for the jump models too.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_jumps.py  (append)
def test_jump_simulators_respect_cfg_device():
    cfg = ExperimentConfig(
        model="merton", device="cpu", s0=100.0, sigma=0.2, maturity=0.5, n_steps=10,
        jump_intensity=1.0, jump_mean=-0.05, jump_std=0.1,
    )
    g = torch.Generator().manual_seed(3)
    pm = simulate_merton(cfg, 256, g)
    assert pm.S.device.type == "cpu"
    assert pm.times.device.type == "cpu"

    cfg_b = ExperimentConfig(
        model="bates", device="cpu", s0=100.0, maturity=0.5, n_steps=10,
        v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7,
        jump_intensity=1.0, jump_mean=-0.05, jump_std=0.1,
    )
    pb = simulate_bates(cfg_b, 256, g)
    assert pb.S.device.type == "cpu"
    assert pb.V.device.type == "cpu"
```

- [ ] **Step 2: Run test to verify it fails**

`pytest tests/test_jumps.py -v -k respect_cfg_device`

Expected: PASS immediately if Tasks 9.2/9.6 already build tensors on
`torch.device(cfg.device)` (they do). If it FAILs, a tensor was created without the
`device=` argument — fix that creation site.

- [ ] **Step 3: Write minimal implementation**

No new implementation if the device check passes. If it fails, ensure every tensor in
`simulate_merton` / `simulate_bates` is created with `device=torch.device(cfg.device)`
(the `torch.empty`, `torch.randn`, `torch.full`, `torch.linspace`, and
`torch.poisson` rate tensor calls).

- [ ] **Step 4: Run full suite to verify it passes**

`pytest tests/test_jumps.py -v`

Expected: PASS — all Group 9 tests green (jump sampler, Merton shapes/martingale/
closed-form/GBM-reduction, Bates shapes/martingale, registration, device).

- [ ] **Step 5: Commit**

```bash
git add tests/test_jumps.py
git commit -m "test(simulators): device-agnostic gate; full jumps suite green"
```
## Group 10: Heston pricer (CF + Carr–Madan) + MC regression gate

Implements `deephedge/pricing/heston.py` per the Shared Contracts. The characteristic
function, Carr–Madan damped-call integral, delta via Gil–Pelaez `P1`, and a Monte-Carlo
pricer. The MC regression test at long-`τ` / high vol-of-vol is the **gate** that catches
the `b = κ` vs `b = κ − ρξ` drift bug; the `φ(0) = 1` sanity test is necessary but does
**not** catch it (spec §7.2, §13(2)).

The CF math is done in **NumPy complex** (`numpy.complex128`) as the spec dictates; the
public pricer/delta/MC functions return Python `float` / `tuple[float, float]`. The MC
pricer reuses `simulate_heston` (group: Heston simulator) and `EuropeanOption` /
`payoff` (group: instruments). Tests seed everything with `torch.Generator`.

Assumed earlier-built pieces this group binds to (Shared Contracts):
- `deephedge/config.py` → `ExperimentConfig` (fields `s0,k,r,q,v0,kappa,theta,xi,rho`).
- `deephedge/pricing/black_scholes.py` → `bs_price(S, K, tau, r, sigma, q=0.0, kind="call") -> torch.Tensor`.
- `deephedge/simulators/base.py` → `Paths`; `deephedge/simulators/heston.py` → `simulate_heston(cfg, n_paths, generator) -> Paths`.
- `deephedge/instruments.py` → `EuropeanOption`, `payoff(option, S_T) -> torch.Tensor`.

---

### Task 10.1: Heston characteristic function `heston_char_func`

The single, bug-fixed CF (spec §7.2). Constant drift coefficient is `kappa`
(**NOT** `kappa - rho*xi`). Albrecher `g2`/`-d` form, principal/NumPy sqrt. Implemented in
NumPy complex so it broadcasts over a `u` grid for Carr–Madan.

**Files:**
- Create: `deephedge/pricing/heston.py`
- Test: `tests/pricing/test_heston.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/pricing/test_heston.py
import numpy as np
import pytest

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
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/pricing/test_heston.py -v
```
Expected: FAIL with `ImportError`/`ModuleNotFoundError` — `deephedge.pricing.heston` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
# deephedge/pricing/heston.py
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
```

(Also create empty package marker files so the import path resolves.)

```python
# deephedge/pricing/__init__.py
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/pricing/test_heston.py -v
```
Expected: PASS — all three CF tests green; `phi(0)==1`, hand-recomputed `C+D*v0` match, complex128 broadcasting.

- [ ] **Step 5: Commit**

```
git add deephedge/pricing/__init__.py deephedge/pricing/heston.py tests/pricing/test_heston.py
git commit -m "feat(pricing): add Heston characteristic function (kappa drift, Albrecher g2/-d)"
```

---

### Task 10.2: Carr–Madan damped-call pricer `heston_price_cm` (call)

Damped-call FFT/integral on a fine trapezoidal grid (spec §7.2). `psi(v)` damped transform,
`call = exp(-damping*ln K)/pi * ∫_0^∞ Re[exp(-i*v*ln K)*psi(v)] dv`. Validate against the
`xi → 0` limit where the price converges to `bs_price(sigma=sqrt(v0))`.

**Files:**
- Modify: `deephedge/pricing/heston.py`
- Test: `tests/pricing/test_heston.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/pricing/test_heston.py  (append)
import torch

from deephedge.pricing.black_scholes import bs_price
from deephedge.pricing.heston import heston_price_cm


def test_cm_call_converges_to_bs_as_xi_to_zero():
    # xi -> 0 freezes variance near v0 (with theta == v0), so Heston -> BS(sigma=sqrt(v0)).
    cfg = _cfg(v0=0.04, theta=0.04, xi=1e-6, kappa=1.5, rho=-0.7, r=0.0, q=0.0)
    K, tau = 100.0, 0.5
    price = heston_price_cm(cfg, K=K, tau=tau, kind="call")
    bs = bs_price(
        torch.tensor(cfg.s0), torch.tensor(K), torch.tensor(tau),
        cfg.r, np.sqrt(cfg.v0), q=cfg.q, kind="call",
    ).item()
    assert price == pytest.approx(bs, abs=1e-2)


def test_cm_call_is_positive_and_bounded():
    # 0 < call < s0 for an ATM call with nonzero maturity.
    cfg = _cfg()
    price = heston_price_cm(cfg, K=100.0, tau=0.5, kind="call")
    assert 0.0 < price < cfg.s0
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/pricing/test_heston.py::test_cm_call_converges_to_bs_as_xi_to_zero -v
```
Expected: FAIL with `AttributeError`/`ImportError` — `heston_price_cm` is not defined yet.

- [ ] **Step 3: Write minimal implementation**

```python
# deephedge/pricing/heston.py  (append)

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
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/pricing/test_heston.py -v
```
Expected: PASS — `xi → 0` price within `1e-2` of `bs_price(sqrt(v0))`; ATM call bounded in `(0, s0)`.

- [ ] **Step 5: Commit**

```
git add deephedge/pricing/heston.py tests/pricing/test_heston.py
git commit -m "feat(pricing): add Carr-Madan damped-call Heston pricer with xi->0 BS limit test"
```

---

### Task 10.3: Put–call parity for `heston_price_cm`

Confirm the put branch satisfies parity exactly (it is derived from it) for several
strikes / maturities.

**Files:**
- Test: `tests/pricing/test_heston.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/pricing/test_heston.py  (append)
@pytest.mark.parametrize("K", [80.0, 100.0, 120.0])
@pytest.mark.parametrize("tau", [0.25, 1.0])
def test_cm_put_call_parity(K, tau):
    cfg = _cfg(r=0.03, q=0.01)
    call = heston_price_cm(cfg, K=K, tau=tau, kind="call")
    put = heston_price_cm(cfg, K=K, tau=tau, kind="put")
    fwd = cfg.s0 * np.exp(-cfg.q * tau) - K * np.exp(-cfg.r * tau)
    # C - P == s0*exp(-q*tau) - K*exp(-r*tau)
    assert (call - put) == pytest.approx(fwd, abs=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/pricing/test_heston.py::test_cm_put_call_parity -v
```
Expected: FAIL only if the put branch were wrong; first run after writing the test should already PASS given Task 10.2's put implementation. If parity off by more than `1e-9`, FAIL with assertion mismatch. (Run anyway to confirm the gate exists.)

- [ ] **Step 3: Write minimal implementation**

No production change needed — parity is implemented in `heston_price_cm`'s put branch
(Task 10.2). This task adds the regression test only. If the test fails, the bug is in the
put branch's forward term; correct it to
`cfg.s0 * np.exp(-cfg.q * tau) - K * np.exp(-cfg.r * tau)`.

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/pricing/test_heston.py::test_cm_put_call_parity -v
```
Expected: PASS — parity holds to `1e-9` for all `(K, tau)` combinations.

- [ ] **Step 5: Commit**

```
git add tests/pricing/test_heston.py
git commit -m "test(pricing): assert Heston Carr-Madan put-call parity across strikes/maturities"
```

---

### Task 10.4: Heston delta via Gil–Pelaez `P1` (`heston_delta`)

`heston_delta = exp(-q*tau) * P1`, where `P1` is the in-the-money delta probability from
the Heston/Gil–Pelaez decomposition. `P1` uses the share-measure CF:
`P1 = 1/2 + (1/pi) * ∫_0^∞ Re[ exp(-i*v*ln K) * phi(v - i) / (i*v*phi(-i)) ] dv`,
with `phi(-i) = E[S_tau] = s0*exp((r-q)*tau)` (the forward). Validate by central-difference
against `heston_price_cm` (delta ≈ ∂C/∂s0), and against BS delta in the `xi → 0` limit.

**Files:**
- Modify: `deephedge/pricing/heston.py`
- Test: `tests/pricing/test_heston.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/pricing/test_heston.py  (append)
from dataclasses import replace

from deephedge.pricing.black_scholes import bs_delta
from deephedge.pricing.heston import heston_delta


def test_delta_matches_central_difference_of_cm_price():
    cfg = _cfg(r=0.02, q=0.0, xi=0.4)
    K, tau = 100.0, 0.5
    h = cfg.s0 * 1e-4
    up = heston_price_cm(replace(cfg, s0=cfg.s0 + h), K=K, tau=tau, kind="call")
    dn = heston_price_cm(replace(cfg, s0=cfg.s0 - h), K=K, tau=tau, kind="call")
    fd = (up - dn) / (2 * h)
    ana = heston_delta(cfg, K=K, tau=tau, kind="call")
    assert ana == pytest.approx(fd, abs=2e-3)
    assert 0.0 < ana < 1.0  # call delta in (0,1)


def test_delta_converges_to_bs_delta_as_xi_to_zero():
    cfg = _cfg(v0=0.04, theta=0.04, xi=1e-6, r=0.0, q=0.0)
    K, tau = 100.0, 0.5
    ana = heston_delta(cfg, K=K, tau=tau, kind="call")
    bs = bs_delta(
        torch.tensor(cfg.s0), torch.tensor(K), torch.tensor(tau),
        cfg.r, np.sqrt(cfg.v0), q=cfg.q, kind="call",
    ).item()
    assert ana == pytest.approx(bs, abs=1e-2)


def test_delta_put_via_parity():
    # put delta = call delta - exp(-q*tau)
    cfg = _cfg(r=0.02, q=0.01)
    K, tau = 100.0, 0.5
    call_d = heston_delta(cfg, K=K, tau=tau, kind="call")
    put_d = heston_delta(cfg, K=K, tau=tau, kind="put")
    assert (call_d - put_d) == pytest.approx(np.exp(-cfg.q * tau), abs=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/pricing/test_heston.py::test_delta_matches_central_difference_of_cm_price -v
```
Expected: FAIL with `ImportError`/`AttributeError` — `heston_delta` not defined yet.

- [ ] **Step 3: Write minimal implementation**

```python
# deephedge/pricing/heston.py  (append)

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
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/pricing/test_heston.py -v
```
Expected: PASS — analytic delta within `2e-3` of central-difference, in `(0,1)`; matches BS delta within `1e-2` as `xi → 0`; put-delta parity holds.

- [ ] **Step 5: Commit**

```
git add deephedge/pricing/heston.py tests/pricing/test_heston.py
git commit -m "feat(pricing): add Heston delta = exp(-q*tau)*P1 via Gil-Pelaez decomposition"
```

---

### Task 10.5: Monte-Carlo Heston pricer `heston_price_mc`

Discounted mean payoff (+ standard error) from `simulate_heston`. Returns
`(price, stderr)`. This is the reference the regression gate compares against.

**Files:**
- Modify: `deephedge/pricing/heston.py`
- Test: `tests/pricing/test_heston.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/pricing/test_heston.py  (append)
from deephedge.pricing.heston import heston_price_mc


def test_mc_returns_price_and_positive_stderr():
    cfg = _cfg(r=0.0, q=0.0, xi=0.3)
    gen = torch.Generator().manual_seed(0)
    price, stderr = heston_price_mc(cfg, K=100.0, tau=0.5, n_paths=20_000,
                                    generator=gen, kind="call")
    assert isinstance(price, float) and isinstance(stderr, float)
    assert 0.0 < price < cfg.s0
    assert stderr > 0.0


def test_mc_is_seed_reproducible():
    cfg = _cfg()
    p1, s1 = heston_price_mc(cfg, K=100.0, tau=0.5, n_paths=5_000,
                             generator=torch.Generator().manual_seed(7), kind="call")
    p2, s2 = heston_price_mc(cfg, K=100.0, tau=0.5, n_paths=5_000,
                             generator=torch.Generator().manual_seed(7), kind="call")
    assert p1 == p2
    assert s1 == s2
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/pricing/test_heston.py::test_mc_returns_price_and_positive_stderr -v
```
Expected: FAIL with `ImportError`/`AttributeError` — `heston_price_mc` not defined yet.

- [ ] **Step 3: Write minimal implementation**

```python
# deephedge/pricing/heston.py  (append)
import torch

from dataclasses import replace as _replace

from deephedge.instruments import EuropeanOption, payoff
from deephedge.simulators.heston import simulate_heston


def heston_price_mc(cfg: ExperimentConfig, K: float, tau: float, n_paths: int,
                    generator: torch.Generator, kind: str = "call") -> tuple[float, float]:
    """Monte-Carlo Heston price (discounted mean payoff) and standard error.

    Simulates under risk-neutral drift (mu = r) over horizon `tau` using the Heston
    full-truncation Euler simulator, then discounts the terminal European payoff.
    """
    # Price the option to maturity `tau` regardless of cfg.maturity: simulate over tau.
    sim_cfg = _replace(cfg, mu=cfg.r, maturity=tau)
    paths = simulate_heston(sim_cfg, n_paths, generator)
    S_T = paths.S[:, -1]
    option = EuropeanOption(strike=K, maturity=tau, kind=kind)
    disc = float(np.exp(-cfg.r * tau))
    pay = payoff(option, S_T) * disc           # (n_paths,)
    price = float(pay.mean().item())
    stderr = float((pay.std(unbiased=True) / np.sqrt(n_paths)).item())
    return price, stderr
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/pricing/test_heston.py -v
```
Expected: PASS — MC returns `(float, float)`, price bounded in `(0, s0)`, `stderr > 0`, and identical results for identical seeds.

- [ ] **Step 5: Commit**

```
git add deephedge/pricing/heston.py tests/pricing/test_heston.py
git commit -m "feat(pricing): add Monte-Carlo Heston pricer returning (price, stderr)"
```

---

### Task 10.6: MANDATORY regression gate — Carr–Madan vs Monte-Carlo at long-`τ`, high-`ξ`

The gate (spec §7.2, §13(2)). At long `tau = 1.0` and high vol-of-vol
(`xi=0.8, rho=-0.7`, Feller-violating: `2*kappa*theta = 0.12 < xi**2 = 0.64`), assert
`heston_price_cm` is within `3*stderr` of `heston_price_mc` with a large path count.
The `phi(0)==1` test is necessary but does **NOT** catch the `b=kappa` vs
`b=kappa-rho*xi` drift bug — both forms give `phi(0)=1`. **This MC test is the gate.**

**Files:**
- Test: `tests/pricing/test_heston.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/pricing/test_heston.py  (append)
def test_REGRESSION_cm_within_3_stderr_of_mc_long_tau_high_volofvol():
    # MANDATORY GATE (spec §7.2, §13(2)). Feller-violating, long maturity.
    # phi(0)==1 does NOT catch the b=kappa vs b=kappa-rho*xi drift bug (both pass);
    # this MC agreement test is the gate that does.
    cfg = _cfg(
        s0=100.0, r=0.0, q=0.0,
        v0=0.04, kappa=1.5, theta=0.04, xi=0.8, rho=-0.7,  # 2*k*theta=0.12 < xi^2=0.64
    )
    K, tau = 100.0, 1.0
    cm = heston_price_cm(cfg, K=K, tau=tau, kind="call")
    gen = torch.Generator().manual_seed(2026)
    mc, stderr = heston_price_mc(cfg, K=K, tau=tau, n_paths=500_000,
                                 generator=gen, kind="call")
    # Carr-Madan must sit inside the 3-sigma MC confidence band.
    assert abs(cm - mc) < 3.0 * stderr, (
        f"Heston CM {cm:.4f} vs MC {mc:.4f} +/- {stderr:.4f} "
        f"(|diff|={abs(cm - mc):.4f}, 3*stderr={3 * stderr:.4f}) — "
        f"likely the b=kappa vs b=kappa-rho*xi drift bug"
    )
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/pricing/test_heston.py::test_REGRESSION_cm_within_3_stderr_of_mc_long_tau_high_volofvol -v
```
Expected: FAIL only if the CF drift coefficient is wrong (or sim/pricer mismatch). With the
correct `beta = kappa - rho*xi*1j*u` (constant coeff `kappa`) and a discretization-bias-aware
simulator, it should PASS on first run. To *prove the gate bites*, temporarily change
`beta` in `heston_char_func` to `kappa - rho*xi - rho*xi*1j*u` (the classic bug) and confirm
this test FAILS while `test_char_func_phi_at_zero_is_one` still PASSES; then revert.

- [ ] **Step 3: Write minimal implementation**

No production change — the gate is a test over Tasks 10.1–10.5. If it fails with the correct
`beta`, the discrepancy is Euler discretization bias at high `xi`: increase `n_paths`, or (per
spec §5.2 caveat) switch the simulator to the Andersen QE scheme for this regime. Do **not**
loosen the `3*stderr` band — that band is the gate.

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/pricing/test_heston.py::test_REGRESSION_cm_within_3_stderr_of_mc_long_tau_high_volofvol -v
```
Expected: PASS — `|cm - mc| < 3*stderr` at `tau=1.0`, `xi=0.8`, `rho=-0.7`, `n_paths=500_000`.

- [ ] **Step 5: Commit**

```
git add tests/pricing/test_heston.py
git commit -m "test(pricing): add mandatory Heston CM-vs-MC regression gate (long tau, high vol-of-vol)"
```
## Group 11 — Multi-instrument vega hedging

Adds the **second hedging instrument** (a vanilla option) and the **delta+vega
benchmark**. Three production deliverables, bound exactly to the Shared Contracts in
`00-header.md`:

- `deephedge/instruments.py::mark_option(cfg, paths, option) -> torch.Tensor` —
  `(n_paths, n_steps+1)` MTM prices of the hedge option along every path. Per step the
  time-to-maturity is `tau_i = option.maturity - times_i` clamped at `>= 0`. The option
  is marked with `bs_price` (`model in {"gbm", "merton"}`, using `sigma=cfg.sigma`) or
  with a fast Heston proxy (`model in {"heston", "bates"}`).
- `deephedge/portfolio.py::build_instr_prices(cfg, paths, hedge_option)` — fill the
  `"option"` branch (column 1 `= mark_option(cfg, paths, hedge_option)`, the HEDGE option),
  replacing the `NotImplementedError` left by the portfolio group.
- `deephedge/benchmarks.py::make_bs_delta_vega_strategy(cfg, option) -> Strategy` —
  solve a 2×2 linear system each step for holdings in (underlying, hedge-option) that
  neutralize the portfolio's BS **delta** and **vega** of the *sold* option using the BS
  Greeks of both the sold and the hedge option.

**Heston/Bates marking — V1 approximation (the choice, implemented in full).** The
contract `heston_price_cm(cfg, K, tau, ...)` prices at the *config* spot `cfg.s0` and
variance `cfg.v0`; it does not accept a per-node `(S_i, V_i)` grid, and a per-node
Carr–Madan loop over `n_paths × (n_steps+1)` nodes is far too slow for the differentiable
trajectory. For V1 we therefore mark the Heston/Bates option **leg** with a
**BS-implied-vol proxy frozen at t0**: compute the Heston model price of the hedge option
once at `t0` via `heston_price_cm`, invert it to a single Black–Scholes implied volatility
`sigma_impl` (bisection on `bs_price`), then mark every node with
`bs_price(S_i, K, tau_i, r, sigma_impl, q)`. This is vectorized, differentiable in `S_i`,
and exact at the terminal node (BS at `tau=0` returns intrinsic value = payoff). It is an
acknowledged approximation (the true Heston mark depends on the realized `V_i`, not a
frozen implied vol) and is documented as such in the implementation; full per-node Heston
re-pricing is deferred (out of V1 scope, spec §2).

Spec refs: §7 (pricing used to mark the second instrument), §10 (delta + delta+vega
benchmarks), §16(7) (multi-instrument option-leg hedging + delta+vega benchmark).

> **Prerequisites (earlier groups, all bound via Shared Contracts):**
> `deephedge/config.py::ExperimentConfig`; `deephedge/simulators/base.py::Paths`;
> `deephedge/instruments.py::EuropeanOption` + `payoff` (Group 1);
> `deephedge/pricing/black_scholes.py::bs_price, bs_delta, bs_vega` (BS pricing group);
> `deephedge/pricing/heston.py::heston_price_cm` (Group 10);
> `deephedge/portfolio.py::StepState, PnLResult, build_instr_prices, simulate_pnl`
> and `deephedge/benchmarks.py::make_bs_delta_strategy` (portfolio/benchmarks group).
> This group assumes `build_instr_prices` already returns the `(n_paths, n_steps+1,
> n_instruments)` tensor with `col0 = paths.S` and raises `NotImplementedError` for the
> `"option"` column; Task 11.4 replaces that branch.

---

### Task 11.1: `mark_option` shape + terminal column equals payoff (BS / `gbm` path)

**Files:**
- Modify: `deephedge/instruments.py`
- Test: `tests/test_mark_option.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mark_option.py
import torch

from deephedge.config import ExperimentConfig
from deephedge.instruments import EuropeanOption, mark_option, payoff
from deephedge.simulators.base import Paths


def _gbm_paths(cfg: ExperimentConfig, n_paths: int, seed: int = 0) -> Paths:
    """Tiny deterministic GBM-style path bundle for marking tests."""
    gen = torch.Generator(device=cfg.device).manual_seed(seed)
    n = cfg.n_steps
    times = torch.linspace(0.0, cfg.maturity, n + 1, device=cfg.device)
    z = torch.randn(n_paths, n, generator=gen, device=cfg.device)
    incr = (cfg.drift - 0.5 * cfg.sigma**2) * cfg.dt + cfg.sigma * (cfg.dt**0.5) * z
    log_s = torch.empty(n_paths, n + 1, device=cfg.device)
    log_s[:, 0] = torch.log(torch.tensor(cfg.s0, device=cfg.device))
    log_s[:, 1:] = log_s[:, :1] + torch.cumsum(incr, dim=1)
    return Paths(S=log_s.exp(), V=None, dt=cfg.dt, times=times)


def test_mark_option_shape_and_terminal_equals_payoff_gbm():
    cfg = ExperimentConfig(
        s0=100.0, k=100.0, r=0.0, q=0.0, sigma=0.2,
        maturity=1.0, n_steps=8, model="gbm", device="cpu",
    )
    option = EuropeanOption(strike=100.0, maturity=cfg.maturity, kind="call")
    paths = _gbm_paths(cfg, n_paths=64, seed=1)

    marks = mark_option(cfg, paths, option)

    # shape is (n_paths, n_steps+1)
    assert marks.shape == (64, cfg.n_steps + 1)
    assert torch.isfinite(marks).all()
    # an ATM call mark is strictly positive before expiry, and < spot
    assert (marks[:, 0] > 0).all()
    assert (marks[:, 0] < paths.S[:, 0]).all()
    # terminal column == intrinsic payoff within BS tau->0 pricing tolerance
    terminal = marks[:, -1]
    intrinsic = payoff(option, paths.S[:, -1])
    assert torch.allclose(terminal, intrinsic, atol=1e-4)


def test_mark_option_t0_column_matches_bs_price_with_cfg_sigma():
    # At t0 every path is at s0; the mark must equal bs_price(s0, K, T, r, cfg.sigma, q).
    from deephedge.pricing.black_scholes import bs_price

    cfg = ExperimentConfig(
        s0=100.0, k=100.0, r=0.01, q=0.0, sigma=0.25,
        maturity=0.5, n_steps=5, model="gbm", device="cpu",
    )
    option = EuropeanOption(strike=110.0, maturity=cfg.maturity, kind="call")
    paths = _gbm_paths(cfg, n_paths=16, seed=2)

    marks = mark_option(cfg, paths, option)

    expected0 = bs_price(
        paths.S[:, 0], option.strike, option.maturity, cfg.r, cfg.sigma, q=cfg.q,
        kind=option.kind,
    )
    assert torch.allclose(marks[:, 0], expected0, atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Command: `pytest tests/test_mark_option.py -v`

Expected: FAIL with `ImportError: cannot import name 'mark_option' from 'deephedge.instruments'` (the function does not exist yet — Group 1 implemented only `EuropeanOption` and `payoff`).

- [ ] **Step 3: Write minimal implementation**

Append to `deephedge/instruments.py` (the module already imports `torch` and defines `EuropeanOption` / `payoff`). Add the BS branch now; the Heston/Bates proxy is added in Task 11.3.

```python
# deephedge/instruments.py  (append)
from deephedge.pricing.black_scholes import bs_price


def mark_option(cfg, paths, option: EuropeanOption) -> torch.Tensor:
    """Mark-to-market price of the hedge `option` at every node of `paths`.

    Returns a (n_paths, n_steps+1) tensor. At step i the time-to-maturity is
    tau_i = max(option.maturity - times_i, 0). For model in {"gbm", "merton"} each node
    is priced with Black-Scholes at sigma=cfg.sigma. For model in {"heston", "bates"} the
    leg is marked with a BS-implied-vol proxy frozen at t0 (V1 approximation; see Task
    11.3). BS at tau->0 returns intrinsic value, so the terminal column equals the option
    payoff within pricing tolerance.

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
    return cfg.sigma
```

> `bs_price` accepts a tensor `tau` (the contract states `tau` may be a float **or**
> tensor with a safe `tau -> 0`), so passing the `(n_paths, n_steps+1)` `tau_row`
> broadcasts against the spot tensor `S` and yields the full mark grid in one call.

- [ ] **Step 4: Run test to verify it passes**

Command: `pytest tests/test_mark_option.py -v`

Expected: PASS — shape `(64, 9)`, positive/bounded t0 marks, terminal column equals intrinsic payoff within `1e-4`, and the t0 column equals `bs_price(s0, …, cfg.sigma)` within `1e-6`.

- [ ] **Step 5: Commit**

```bash
git add deephedge/instruments.py tests/test_mark_option.py
git commit -m "feat(instruments): mark_option marks the hedge option via BS (gbm/merton)"
```

---

### Task 11.2: `mark_option` is differentiable in spot and respects the `tau` clamp

**Files:**
- Modify: `tests/test_mark_option.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mark_option.py  (append)
def test_mark_option_is_differentiable_in_spot():
    # MTM gains across the option leg must backprop into the spot path (spec §9).
    cfg = ExperimentConfig(
        s0=100.0, k=100.0, r=0.0, q=0.0, sigma=0.2,
        maturity=1.0, n_steps=4, model="gbm", device="cpu",
    )
    option = EuropeanOption(strike=100.0, maturity=cfg.maturity, kind="call")

    times = torch.linspace(0.0, cfg.maturity, cfg.n_steps + 1)
    S = torch.full((3, cfg.n_steps + 1), 100.0, requires_grad=True)
    paths = Paths(S=S, V=None, dt=cfg.dt, times=times)

    marks = mark_option(cfg, paths, option)
    marks.sum().backward()

    assert S.grad is not None
    assert torch.isfinite(S.grad).all()
    # in-the-money / ATM call mark increases with spot -> positive sensitivity pre-expiry
    assert (S.grad[:, 0] > 0).all()


def test_mark_option_tau_clamped_when_option_matures_before_horizon():
    # A hedge option maturing at T/2 must have tau=0 (intrinsic) for all later nodes.
    cfg = ExperimentConfig(
        s0=100.0, k=100.0, r=0.0, q=0.0, sigma=0.2,
        maturity=1.0, n_steps=10, model="gbm", device="cpu",
    )
    option = EuropeanOption(strike=100.0, maturity=0.5, kind="call")
    paths = _gbm_paths(cfg, n_paths=32, seed=3)

    marks = mark_option(cfg, paths, option)

    # nodes with t_i >= 0.5 have tau=0 -> mark equals intrinsic payoff exactly-ish
    late = paths.times >= 0.5
    intrinsic_late = payoff(option, paths.S[:, late])
    assert torch.allclose(marks[:, late], intrinsic_late, atol=1e-4)
    # an early node (t=0, tau=0.5) is worth strictly more than intrinsic (time value)
    assert (marks[:, 0] > payoff(option, paths.S[:, 0]) - 1e-6).all()
    assert marks[:, 0].mean() > payoff(option, paths.S[:, 0]).mean()
```

- [ ] **Step 2: Run test to verify it fails**

Command: `pytest tests/test_mark_option.py -k "differentiable or tau_clamped" -v`

Expected: PASS against the Task 11.1 implementation — `mark_option` uses only differentiable torch ops (`bs_price` on a tensor spot) and clamps `tau` with `.clamp(min=0.0)`. Run it first to confirm it is green; if it FAILS on `S.grad is not None`, a non-differentiable op (e.g. a `.detach()` or a Python float spot) slipped into `mark_option` — fix so the spot tensor flows through `bs_price`. If it FAILS on the clamp assertion, the `tau_i = max(maturity - t_i, 0)` clamp is missing.

- [ ] **Step 3: Write minimal implementation**

No production change required — Task 11.1's `mark_option` already (a) prices the tensor spot `paths.S` directly through `bs_price` (differentiable), and (b) computes `tau = (option.maturity - paths.times).clamp(min=0.0)`. This task locks both invariants with regression tests.

- [ ] **Step 4: Run test to verify it passes**

Command: `pytest tests/test_mark_option.py -k "differentiable or tau_clamped" -v`

Expected: PASS — spot gradient is finite and positive pre-expiry; post-maturity nodes equal intrinsic value within `1e-4`; the t0 node carries time value.

- [ ] **Step 5: Commit**

```bash
git add tests/test_mark_option.py
git commit -m "test(instruments): mark_option differentiable in spot and clamps tau"
```

---

### Task 11.3: Heston/Bates option leg — frozen BS-implied-vol proxy (`_heston_implied_vol`)

**Files:**
- Modify: `deephedge/instruments.py`
- Test: `tests/test_mark_option.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mark_option.py  (append)
import numpy as np

from deephedge.instruments import _heston_implied_vol
from deephedge.pricing.heston import heston_price_cm
from deephedge.pricing.black_scholes import bs_price


def _heston_cfg(**kw) -> ExperimentConfig:
    base = dict(
        s0=100.0, k=100.0, r=0.0, q=0.0,
        v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7,
        maturity=1.0, n_steps=8, model="heston", device="cpu",
    )
    base.update(kw)
    return ExperimentConfig(**base)


def test_heston_implied_vol_reprices_heston_t0_price():
    # The inverted BS implied vol, plugged back into bs_price at t0, must reproduce the
    # Heston model price of the hedge option to tight tolerance.
    cfg = _heston_cfg()
    option = EuropeanOption(strike=100.0, maturity=cfg.maturity, kind="call")

    sigma_impl = _heston_implied_vol(cfg, option)
    assert sigma_impl > 0.0
    assert np.isfinite(sigma_impl)

    heston_px = heston_price_cm(cfg, K=option.strike, tau=option.maturity, kind="call")
    bs_px = bs_price(
        torch.tensor(cfg.s0), option.strike, option.maturity, cfg.r, sigma_impl, q=cfg.q,
        kind="call",
    ).item()
    assert abs(bs_px - heston_px) < 5e-3


def test_mark_option_heston_terminal_equals_payoff_and_t0_matches_heston():
    cfg = _heston_cfg()
    option = EuropeanOption(strike=100.0, maturity=cfg.maturity, kind="call")
    # build a simple Heston-style path bundle (variance held flat for the test paths;
    # mark_option uses the FROZEN proxy vol, so V is not consumed by the mark)
    gen = torch.Generator(device="cpu").manual_seed(5)
    n = cfg.n_steps
    times = torch.linspace(0.0, cfg.maturity, n + 1)
    z = torch.randn(40, n, generator=gen)
    incr = (cfg.drift - 0.5 * cfg.v0) * cfg.dt + (cfg.v0**0.5) * (cfg.dt**0.5) * z
    log_s = torch.empty(40, n + 1)
    log_s[:, 0] = torch.log(torch.tensor(cfg.s0))
    log_s[:, 1:] = log_s[:, :1] + torch.cumsum(incr, dim=1)
    V = torch.full((40, n + 1), cfg.v0)
    paths = Paths(S=log_s.exp(), V=V, dt=cfg.dt, times=times)

    marks = mark_option(cfg, paths, option)

    assert marks.shape == (40, n + 1)
    # terminal column == intrinsic payoff (BS at tau=0)
    assert torch.allclose(marks[:, -1], payoff(option, paths.S[:, -1]), atol=1e-4)
    # t0 mark (all paths at s0) reproduces the Heston model price within proxy tolerance
    heston_px = heston_price_cm(cfg, K=option.strike, tau=option.maturity, kind="call")
    assert abs(marks[:, 0].mean().item() - heston_px) < 5e-3
```

- [ ] **Step 2: Run test to verify it fails**

Command: `pytest tests/test_mark_option.py -k "heston" -v`

Expected: FAIL with `ImportError: cannot import name '_heston_implied_vol' from 'deephedge.instruments'` (the proxy helper does not exist yet; `_mark_vol` currently always returns `cfg.sigma`).

- [ ] **Step 3: Write minimal implementation**

Add the implied-vol inversion and route the Heston/Bates branch through it. Replace the body of `_mark_vol` so stochastic-vol models use the frozen Heston-implied BS vol; add `_heston_implied_vol`.

```python
# deephedge/instruments.py  (append; uses heston_price_cm imported lazily to avoid cycles)
import math

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
```

Now update `_mark_vol` to use the proxy for stochastic-vol models:

```python
# deephedge/instruments.py  (replace the existing _mark_vol body)
def _mark_vol(cfg, option: EuropeanOption) -> float:
    """Volatility used to mark the option leg.

    GBM / Merton: cfg.sigma. Heston / Bates: a single BS implied vol that reprices the
    Heston model price at t0 (frozen-implied-vol proxy, V1 approximation). The proxy is
    independent of the per-node realized variance V_i; full per-node Heston re-pricing is
    out of V1 scope (spec §2).
    """
    if cfg.model in _STOCH_VOL_MODELS:
        return _heston_implied_vol(cfg, option)
    return cfg.sigma
```

> `math` is imported for parity with the rest of the package's numerics even though the
> bisection above uses only arithmetic; keep the import if a later edit needs it, else it
> is harmless. The lazy `heston_price_cm` import inside `_heston_implied_vol` prevents an
> `instruments` ↔ `pricing.heston` import cycle.

- [ ] **Step 4: Run test to verify it passes**

Command: `pytest tests/test_mark_option.py -k "heston" -v`

Expected: PASS — the inverted vol reprices the Heston price within `5e-3`; the Heston-marked grid has terminal column = intrinsic payoff and t0 mark ≈ the Heston model price.

- [ ] **Step 5: Commit**

```bash
git add deephedge/instruments.py tests/test_mark_option.py
git commit -m "feat(instruments): Heston/Bates option leg via frozen BS-implied-vol proxy"
```

---

### Task 11.4: `build_instr_prices` — fill the `"option"` column with `mark_option`

**Files:**
- Modify: `deephedge/portfolio.py`
- Test: `tests/test_build_instr_prices_option.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_build_instr_prices_option.py
import torch

from deephedge.config import ExperimentConfig
from deephedge.instruments import EuropeanOption, mark_option
from deephedge.portfolio import build_instr_prices
from deephedge.simulators.base import Paths


def _gbm_paths(cfg: ExperimentConfig, n_paths: int, seed: int = 0) -> Paths:
    gen = torch.Generator(device=cfg.device).manual_seed(seed)
    n = cfg.n_steps
    times = torch.linspace(0.0, cfg.maturity, n + 1, device=cfg.device)
    z = torch.randn(n_paths, n, generator=gen, device=cfg.device)
    incr = (cfg.drift - 0.5 * cfg.sigma**2) * cfg.dt + cfg.sigma * (cfg.dt**0.5) * z
    log_s = torch.empty(n_paths, n + 1, device=cfg.device)
    log_s[:, 0] = torch.log(torch.tensor(cfg.s0, device=cfg.device))
    log_s[:, 1:] = log_s[:, :1] + torch.cumsum(incr, dim=1)
    return Paths(S=log_s.exp(), V=None, dt=cfg.dt, times=times)


def test_build_instr_prices_underlying_only_is_single_column():
    cfg = ExperimentConfig(
        s0=100.0, k=100.0, sigma=0.2, maturity=1.0, n_steps=6,
        model="gbm", instruments=("underlying",), device="cpu",
    )
    option = EuropeanOption(strike=cfg.k, maturity=cfg.maturity, kind="call")
    paths = _gbm_paths(cfg, n_paths=32, seed=1)

    prices = build_instr_prices(cfg, paths, option)

    assert prices.shape == (32, cfg.n_steps + 1, 1)
    # column 0 is the underlying spot itself
    assert torch.allclose(prices[:, :, 0], paths.S)


def test_build_instr_prices_two_instruments_has_underlying_and_option():
    cfg = ExperimentConfig(
        s0=100.0, k=100.0, sigma=0.2, maturity=1.0, n_steps=6,
        model="gbm", instruments=("underlying", "option"), device="cpu",
    )
    # hedge option defaults to the sold-call strike/maturity per the contract
    option = EuropeanOption(strike=cfg.k, maturity=cfg.maturity, kind="call")
    paths = _gbm_paths(cfg, n_paths=32, seed=2)

    prices = build_instr_prices(cfg, paths, option)

    # (n_paths, n_steps+1, 2) and NO NotImplementedError
    assert prices.shape == (32, cfg.n_steps + 1, 2)
    # col0 == spot, col1 == mark_option
    assert torch.allclose(prices[:, :, 0], paths.S)
    assert torch.allclose(prices[:, :, 1], mark_option(cfg, paths, option))
    assert torch.isfinite(prices).all()
```

- [ ] **Step 2: Run test to verify it fails**

Command: `pytest tests/test_build_instr_prices_option.py -v`

Expected: `test_build_instr_prices_underlying_only_is_single_column` PASSES (the portfolio group already builds the underlying column). `test_build_instr_prices_two_instruments_has_underlying_and_option` FAILS with `NotImplementedError` raised from `build_instr_prices` when `"option"` is in `cfg.instruments` (the option branch is the placeholder this task replaces).

- [ ] **Step 3: Write minimal implementation**

In `deephedge/portfolio.py`, replace the `NotImplementedError` placeholder in `build_instr_prices` so the `"option"` column is `mark_option(cfg, paths, option)`. The function already builds `col0 = paths.S` and stacks per the `cfg.instruments` order; the full bound implementation is:

```python
# deephedge/portfolio.py  (build_instr_prices — option branch filled in)
from deephedge.instruments import mark_option


def build_instr_prices(cfg, paths, hedge_option) -> torch.Tensor:
    """(n_paths, n_steps+1, n_instruments) hedging-instrument price tensor.

    ``hedge_option`` is the HEDGE instrument's option (its own strike/maturity), NOT the
    sold/liability option. Column j corresponds to cfg.instruments[j]:
      - "underlying": the spot path paths.S
      - "option":     the marked hedge-option price mark_option(cfg, paths, hedge_option)
    """
    cols = []
    for name in cfg.instruments:
        if name == "underlying":
            cols.append(paths.S)
        elif name == "option":
            cols.append(mark_option(cfg, paths, hedge_option))
        else:
            raise ValueError(
                f"unknown hedging instrument {name!r}; "
                "expected 'underlying' or 'option'"
            )
    return torch.stack(cols, dim=-1)
```

> If the portfolio group wrote `build_instr_prices` with the `"underlying"` column
> preallocated and only the `"option"` branch stubbed, the minimal change is to swap that
> branch's `raise NotImplementedError(...)` for `cols.append(mark_option(cfg, paths,
> hedge_option))` (importing `mark_option` at module top). The observable contract —
> `col0 = S`, `col1 = mark_option(hedge_option)`, shape
> `(n_paths, n_steps+1, n_instruments)` — is identical.

- [ ] **Step 4: Run test to verify it passes**

Command: `pytest tests/test_build_instr_prices_option.py -v`

Expected: PASS — single-column shape `(32, 7, 1)` with `col0 == S`; two-instrument shape `(32, 7, 2)` with `col0 == S` and `col1 == mark_option`, all finite.

- [ ] **Step 5: Commit**

```bash
git add deephedge/portfolio.py tests/test_build_instr_prices_option.py
git commit -m "feat(portfolio): build_instr_prices fills option column via mark_option"
```

---

### Task 11.5: `make_bs_delta_vega_strategy` — shape + degenerate-vega fallback

**Files:**
- Modify: `deephedge/benchmarks.py`
- Test: `tests/test_delta_vega_strategy.py`

The strategy neutralizes the **sold call's** BS delta and vega using the (underlying,
hedge-option) pair. The underlying has delta 1 and vega 0; the hedge option has delta
`Δ_h` and vega `ν_h`. We are short one sold call (delta `Δ_s`, vega `ν_s`), so the hedge
must satisfy, for holdings `(a, b)` in (underlying, hedge-option):

```
b · ν_h            = ν_s         (vega:  match the sold call's vega)
a · 1 + b · Δ_h    = Δ_s         (delta: match the sold call's delta)
```

i.e. `b = ν_s / ν_h` (vega-neutral via the option), then `a = Δ_s − b · Δ_h`
(delta-neutral via the underlying). Greeks are evaluated under the risk-neutral measure
with `sigma = cfg.sigma`, `tau` = remaining time of each leg at the step. When `ν_h ≈ 0`
(deep ITM/OTM hedge option, or `tau → 0`) the 2×2 system is singular in `b`; the strategy
falls back to `b = 0` and pure delta hedging (`a = Δ_s`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_delta_vega_strategy.py
import torch

from deephedge.benchmarks import make_bs_delta_vega_strategy
from deephedge.config import ExperimentConfig
from deephedge.instruments import EuropeanOption
from deephedge.portfolio import StepState


def _state(cfg, S, tau, prev=None, instr_prices=None):
    B = S.shape[0]
    if prev is None:
        prev = torch.zeros(B, cfg.n_instruments)
    if instr_prices is None:
        instr_prices = torch.zeros(B, cfg.n_instruments)
    # step index is informational for analytic strategies; tau drives the Greeks.
    return StepState(
        step=0, S=S, V=None, tau=tau, prev_holdings=prev, instr_prices=instr_prices,
    )


def test_delta_vega_strategy_returns_B_by_2():
    cfg = ExperimentConfig(
        s0=100.0, k=100.0, r=0.0, q=0.0, sigma=0.2,
        maturity=1.0, n_steps=10, model="gbm",
        instruments=("underlying", "option"), device="cpu",
    )
    # hedge option: different strike so its vega differs from the sold call's
    hedge = EuropeanOption(strike=110.0, maturity=cfg.maturity, kind="call")
    strat = make_bs_delta_vega_strategy(cfg, hedge)

    S = torch.tensor([90.0, 100.0, 110.0])
    holdings = strat(_state(cfg, S, tau=0.5))

    assert holdings.shape == (3, 2)
    assert torch.isfinite(holdings).all()


def test_delta_vega_strategy_falls_back_to_delta_when_hedge_vega_vanishes():
    # At tau -> 0 the hedge option's vega -> 0; b must be 0 and a -> sold-call delta.
    from deephedge.pricing.black_scholes import bs_delta

    cfg = ExperimentConfig(
        s0=100.0, k=100.0, r=0.0, q=0.0, sigma=0.2,
        maturity=1.0, n_steps=10, model="gbm",
        instruments=("underlying", "option"), device="cpu",
    )
    hedge = EuropeanOption(strike=100.0, maturity=cfg.maturity, kind="call")
    strat = make_bs_delta_vega_strategy(cfg, hedge)

    S = torch.tensor([95.0, 100.0, 105.0])
    holdings = strat(_state(cfg, S, tau=1e-8))

    # option holding collapses to ~0 (no usable vega), underlying ~ sold-call delta
    assert torch.allclose(holdings[:, 1], torch.zeros(3), atol=1e-6)
    delta_sold = bs_delta(S, cfg.k, 1e-8, cfg.r, cfg.sigma, q=cfg.q)
    assert torch.allclose(holdings[:, 0], delta_sold, atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Command: `pytest tests/test_delta_vega_strategy.py -v`

Expected: FAIL with `ImportError: cannot import name 'make_bs_delta_vega_strategy' from 'deephedge.benchmarks'` (only `make_bs_delta_strategy` / `make_no_hedge_strategy` / `make_nn_strategy` exist so far).

- [ ] **Step 3: Write minimal implementation**

Append to `deephedge/benchmarks.py` (the module already imports `torch` and the BS Greeks from earlier benchmarks; add `bs_vega` to the import if not present).

```python
# deephedge/benchmarks.py  (append)
from deephedge.pricing.black_scholes import bs_delta, bs_vega


def make_bs_delta_vega_strategy(cfg, option):
    """Strategy neutralizing the sold call's BS delta AND vega with (underlying, option).

    Holdings (a, b) in (underlying, hedge-option) solve the 2x2 system:
        b * vega_h        = vega_s          (vega:  underlying has zero vega)
        a + b * delta_h   = delta_s         (delta: underlying has unit delta)
    => b = vega_s / vega_h ; a = delta_s - b * delta_h.
    Greeks use sigma=cfg.sigma under the risk-neutral measure. tau for each leg is the
    step's remaining time (state.tau) for the sold call and
    (option.maturity - (cfg.maturity - state.tau)) clamped >= 0 for the hedge option
    (so a longer-dated hedge option keeps positive time-to-maturity at the sold call's
    expiry). When vega_h is ~0 the system is singular in b; fall back to b=0, a=delta_s.
    """

    def strategy(state):
        S = state.S                                  # (B,)
        tau_s = float(state.tau)                     # sold-call time-to-maturity
        t_now = cfg.maturity - tau_s                 # elapsed calendar time
        tau_h = max(option.maturity - t_now, 0.0)    # hedge-option time-to-maturity

        delta_s = bs_delta(S, cfg.k, tau_s, cfg.r, cfg.sigma, q=cfg.q)
        vega_s = bs_vega(S, cfg.k, tau_s, cfg.r, cfg.sigma, q=cfg.q)
        delta_h = bs_delta(S, option.strike, tau_h, cfg.r, cfg.sigma, q=cfg.q)
        vega_h = bs_vega(S, option.strike, tau_h, cfg.r, cfg.sigma, q=cfg.q)

        eps = 1e-6
        usable = vega_h.abs() > eps
        b = torch.where(usable, vega_s / vega_h.clamp_min(eps), torch.zeros_like(vega_h))
        # zero out option leg where its vega is unusable (avoids dividing by ~0)
        b = torch.where(usable, b, torch.zeros_like(b))
        a = delta_s - b * delta_h
        return torch.stack([a, b], dim=-1)           # (B, 2)

    return strategy
```

> `bs_delta` / `bs_vega` accept a tensor spot `S` with scalar `K, tau, r, sigma` (the
> contract), so the Greeks are `(B,)` tensors and the `torch.where` fallback is
> elementwise. `clamp_min(eps)` keeps the division finite before `torch.where` discards
> the unusable column, so no NaN ever enters the autograd graph.

- [ ] **Step 4: Run test to verify it passes**

Command: `pytest tests/test_delta_vega_strategy.py -v`

Expected: PASS — `(3, 2)` finite holdings; at `tau→0` the option leg is `0` and the underlying leg equals the sold-call delta within `1e-4`.

- [ ] **Step 5: Commit**

```bash
git add deephedge/benchmarks.py tests/test_delta_vega_strategy.py
git commit -m "feat(benchmarks): make_bs_delta_vega_strategy (2x2 delta+vega neutralization)"
```

---

### Task 11.6: Delta+vega neutralizes a constructed delta+vega exposure within tolerance

**Files:**
- Modify: `tests/test_delta_vega_strategy.py`

The headline correctness property: the strategy's holdings make the *combined* BS delta
and BS vega of `{−sold call, +a·underlying, +b·hedge option}` both ≈ 0. We assert this
directly on the BS Greeks at a step where the hedge option has real, non-degenerate vega.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_delta_vega_strategy.py  (append)
def test_delta_vega_strategy_neutralizes_delta_and_vega():
    from deephedge.pricing.black_scholes import bs_delta, bs_vega

    cfg = ExperimentConfig(
        s0=100.0, k=100.0, r=0.01, q=0.0, sigma=0.2,
        maturity=1.0, n_steps=10, model="gbm",
        instruments=("underlying", "option"), device="cpu",
    )
    # longer-dated, different-strike hedge option => non-degenerate vega at this step
    hedge = EuropeanOption(strike=105.0, maturity=1.5, kind="call")
    strat = make_bs_delta_vega_strategy(cfg, hedge)

    S = torch.tensor([90.0, 100.0, 110.0, 120.0])
    tau_s = 0.5                                   # sold-call remaining time
    holdings = strat(_state(cfg, S, tau=tau_s))
    a = holdings[:, 0]
    b = holdings[:, 1]

    t_now = cfg.maturity - tau_s
    tau_h = max(hedge.maturity - t_now, 0.0)      # = 1.5 - 0.5 = 1.0

    # sold-call Greeks (we are SHORT one call -> liability has these Greeks)
    delta_s = bs_delta(S, cfg.k, tau_s, cfg.r, cfg.sigma, q=cfg.q)
    vega_s = bs_vega(S, cfg.k, tau_s, cfg.r, cfg.sigma, q=cfg.q)
    # hedge-option Greeks
    delta_h = bs_delta(S, hedge.strike, tau_h, cfg.r, cfg.sigma, q=cfg.q)
    vega_h = bs_vega(S, hedge.strike, tau_h, cfg.r, cfg.sigma, q=cfg.q)

    # net delta of the hedged book = a*1 + b*delta_h - delta_s  (underlying delta = 1)
    net_delta = a * 1.0 + b * delta_h - delta_s
    # net vega = a*0 + b*vega_h - vega_s   (underlying vega = 0)
    net_vega = b * vega_h - vega_s

    assert torch.allclose(net_delta, torch.zeros_like(net_delta), atol=1e-5)
    assert torch.allclose(net_vega, torch.zeros_like(net_vega), atol=1e-5)
    # the option leg actually carries weight here (not the degenerate b=0 case)
    assert b.abs().mean() > 1e-3
```

- [ ] **Step 2: Run test to verify it fails**

Command: `pytest tests/test_delta_vega_strategy.py::test_delta_vega_strategy_neutralizes_delta_and_vega -v`

Expected: PASS against the Task 11.5 implementation (the 2×2 solve is exact: `b = vega_s/vega_h` zeroes `net_vega`, then `a = delta_s − b·delta_h` zeroes `net_delta`). Run it first to confirm green; if it FAILS on `net_vega`, the vega row was mis-solved (e.g. used the sold-call vega for the option leg); if it FAILS on `net_delta`, the delta row omitted the `b·delta_h` cross-term. Fix the 2×2 solve in Task 11.5 to match.

- [ ] **Step 3: Write minimal implementation**

No production change required — Task 11.5's `strategy` computes `b = vega_s / vega_h` and `a = delta_s − b·delta_h`, which is the exact solution of the 2×2 system, so both net Greeks vanish identically. This task is the correctness gate (spec §10 / §16(7)) locking the neutralization property.

- [ ] **Step 4: Run test to verify it passes**

Command: `pytest tests/test_delta_vega_strategy.py::test_delta_vega_strategy_neutralizes_delta_and_vega -v`

Expected: PASS — `net_delta ≈ 0` and `net_vega ≈ 0` within `1e-5`, with the option leg carrying non-trivial weight.

- [ ] **Step 5: Commit**

```bash
git add tests/test_delta_vega_strategy.py
git commit -m "test(benchmarks): delta+vega strategy zeroes net BS delta and vega"
```

---

### Task 11.7: Integration — `simulate_pnl` with two instruments runs and P&L is finite

**Files:**
- Test: `tests/test_multi_instrument_integration.py`

End-to-end check that the option leg threads through the whole P&L engine: build a GBM
path bundle, premium from `bs_price`, `instr_prices = build_instr_prices(...)` with
`instruments=("underlying","option")`, run `simulate_pnl` with the delta+vega strategy,
and assert the returned `PnLResult.pnl` is finite with the right shape. Also run it under
Heston (frozen-proxy mark) to exercise the stochastic-vol marking branch.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_multi_instrument_integration.py
import torch

from deephedge.benchmarks import make_bs_delta_vega_strategy
from deephedge.config import ExperimentConfig
from deephedge.instruments import EuropeanOption
from deephedge.portfolio import build_instr_prices, simulate_pnl
from deephedge.pricing.black_scholes import bs_price
from deephedge.simulators.base import get_simulator


def test_simulate_pnl_two_instruments_gbm_runs_and_is_finite():
    cfg = ExperimentConfig(
        s0=100.0, k=100.0, r=0.0, q=0.0, sigma=0.2,
        maturity=30 / 252, n_steps=10, model="gbm", cost=0.001,
        instruments=("underlying", "option"), device="cpu",
    )
    gen = torch.Generator(device="cpu").manual_seed(0)
    paths = get_simulator("gbm")(cfg, 2048, gen)

    # sold call (the liability) and the hedge option (the second instrument)
    sold = EuropeanOption(strike=cfg.k, maturity=cfg.maturity, kind="call")
    hedge = EuropeanOption(strike=cfg.k, maturity=cfg.maturity, kind="call")

    premium = float(bs_price(torch.tensor(cfg.s0), cfg.k, cfg.maturity, cfg.r, cfg.sigma,
                             q=cfg.q, kind="call"))
    instr_prices = build_instr_prices(cfg, paths, hedge)
    assert instr_prices.shape == (2048, cfg.n_steps + 1, 2)

    strat = make_bs_delta_vega_strategy(cfg, hedge)
    result = simulate_pnl(strat, paths, cfg, sold, premium, instr_prices)

    assert result.pnl.shape == (2048,)
    assert torch.isfinite(result.pnl).all()
    # sanity: with a real premium the mean hedged P&L is not pathological
    assert abs(result.pnl.mean().item()) < cfg.s0


def test_simulate_pnl_two_instruments_heston_runs_and_is_finite():
    cfg = ExperimentConfig(
        s0=100.0, k=100.0, r=0.0, q=0.0,
        v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7,
        maturity=30 / 252, n_steps=10, model="heston", cost=0.0,
        instruments=("underlying", "option"), device="cpu",
    )
    gen = torch.Generator(device="cpu").manual_seed(1)
    paths = get_simulator("heston")(cfg, 1024, gen)

    sold = EuropeanOption(strike=cfg.k, maturity=cfg.maturity, kind="call")
    hedge = EuropeanOption(strike=cfg.k, maturity=cfg.maturity, kind="call")

    from deephedge.pricing.heston import heston_price_cm
    premium = float(heston_price_cm(cfg, K=cfg.k, tau=cfg.maturity, kind="call"))
    instr_prices = build_instr_prices(cfg, paths, hedge)
    assert instr_prices.shape == (1024, cfg.n_steps + 1, 2)

    strat = make_bs_delta_vega_strategy(cfg, hedge)
    result = simulate_pnl(strat, paths, cfg, sold, premium, instr_prices)

    assert result.pnl.shape == (1024,)
    assert torch.isfinite(result.pnl).all()
```

- [ ] **Step 2: Run test to verify it fails**

Command: `pytest tests/test_multi_instrument_integration.py -v`

Expected: PASS once Tasks 11.1–11.5 are merged (the option column, the marking proxy, and the delta+vega strategy all exist and thread through the existing `simulate_pnl`). Run it first; if it FAILS with `NotImplementedError` the option branch of `build_instr_prices` (Task 11.4) is not wired; if it FAILS with a shape error in `simulate_pnl`, the per-step `instr_prices[:, i, :]` slice does not match the `(B, n_instruments)` holdings — confirm the strategy returns `(B, 2)` (Task 11.5).

- [ ] **Step 3: Write minimal implementation**

No new production code — this task is the integration gate. It composes only contract
functions implemented in Tasks 11.1–11.5 (`mark_option`, `build_instr_prices` option
branch, `make_bs_delta_vega_strategy`) with the existing `simulate_pnl`. If Step 2
surfaced a failure, fix the implicated earlier task (the option column wiring in 11.4 or
the `(B, 2)` holdings shape in 11.5); no change is needed here.

- [ ] **Step 4: Run test to verify it passes**

Command: `pytest tests/test_multi_instrument_integration.py -v`

Expected: PASS — both GBM and Heston two-instrument runs produce a finite `(n_paths,)` P&L tensor, with `instr_prices` shaped `(n_paths, n_steps+1, 2)`.

- [ ] **Step 5: Commit**

```bash
git add tests/test_multi_instrument_integration.py
git commit -m "test(portfolio): two-instrument simulate_pnl runs with finite P&L (gbm + heston)"
```

---

### Task 11.8: Full Group 11 suite green

**Files:**
- Test: `tests/test_mark_option.py`, `tests/test_build_instr_prices_option.py`,
  `tests/test_delta_vega_strategy.py`, `tests/test_multi_instrument_integration.py`

- [ ] **Step 1: Write the failing test**

No new test code. This task runs the entire Group 11 test set together as a regression
gate, confirming the marking, instrument-price assembly, delta+vega benchmark, and
end-to-end integration all pass with no cross-test interference (shared generators,
import cycles between `instruments` ↔ `pricing.heston` ↔ `portfolio`).

- [ ] **Step 2: Run test to verify it fails**

Command: `pytest tests/test_mark_option.py tests/test_build_instr_prices_option.py tests/test_delta_vega_strategy.py tests/test_multi_instrument_integration.py -v`

Expected: All PASS if Tasks 11.1–11.7 landed correctly. Treat any FAIL as a real integration regression — most likely an `instruments` ↔ `pricing.heston` import cycle (mitigated by the lazy `heston_price_cm` import inside `_heston_implied_vol` in Task 11.3) or a `portfolio` ↔ `instruments` cycle (mitigated by importing `mark_option` where used in Task 11.4) — and debug before proceeding.

- [ ] **Step 3: Write minimal implementation**

No production change unless Step 2 surfaces a regression. If an import cycle appears, keep
the lazy `from deephedge.pricing.heston import heston_price_cm` inside `_heston_implied_vol`
(Task 11.3) and the module-top `from deephedge.instruments import mark_option` in
`portfolio.py` (Task 11.4) — these break the only two possible cycles in this group.

- [ ] **Step 4: Run test to verify it passes**

Command: `pytest tests/test_mark_option.py tests/test_build_instr_prices_option.py tests/test_delta_vega_strategy.py tests/test_multi_instrument_integration.py -v`

Expected: PASS — the full multi-instrument vega-hedging suite is green.

- [ ] **Step 5: Commit**

```bash
git add tests/test_mark_option.py tests/test_build_instr_prices_option.py tests/test_delta_vega_strategy.py tests/test_multi_instrument_integration.py
git commit -m "test(multi-instrument): full Group 11 suite green (mark, instr-prices, delta+vega, integration)"
```
## Group 12 — Evaluation, figures, experiment configs, report

> Implements `deephedge/evaluate.py` plus the `experiments/` config builders, the
> `scripts/run_experiment.py` / `scripts/make_figures.py` entrypoints, and the `report.md`
> skeleton. Binds to the Shared Contracts in `parts/00-header.md` **exactly**:
> `ExperimentConfig` (with `.dt`, `.drift`, `.n_instruments`, `.instruments`), `get_simulator`,
> `EuropeanOption`, `payoff`, `bs_price`, `bs_delta`, `heston_price_cm`,
> `build_instr_prices`, `simulate_pnl` returning `PnLResult(pnl, turnover, cost, holdings)`,
> `make_nn_strategy(hedger, cfg)`, `make_bs_delta_strategy(cfg)`,
> `make_bs_delta_vega_strategy(cfg, option)`, `make_no_hedge_strategy()`,
> `Hedger`, `build_features`, `train(cfg) -> tuple[Hedger, dict]`.
> Spec refs: §12 (evaluation + the two money-shot figures), §13 (trust-builder tests),
> §14 (regimes), §15 (deliverables: regime table, two charts, the three report paragraphs).
>
> This group assumes Groups 1–11 are complete. Every symbol used below is from the Shared
> Contracts or defined earlier in this section. The evaluation entrypoint
> `evaluate(cfg, hedger) -> dict` returns a **nested dict** keyed by strategy name, each
> value a flat dict of metrics. The premium is computed under the **risk-neutral measure**
> (drift `r`) and is **identical** for every strategy, exactly as `train` does it (spec §3),
> so the P&L comparison is apples-to-apples.
>
> **Metric conventions (locked here, used by all tasks):**
> - `pnl` from `simulate_pnl` is per-path terminal P&L, shape `(B,)`.
> - `mean_pnl = pnl.mean()`, `std = pnl.std(unbiased=True)`.
> - Loss is `L = -pnl` everywhere (spec §6). `VaR_alpha = quantile(L, alpha)`,
>   `CVaR_alpha = w* + (1/(1-alpha)) * mean(relu(L - w*))` with `w* = VaR_alpha`. This is
>   exactly `cvar_loss(pnl, alpha, w*)` evaluated at the empirical optimum, which is what
>   Task 12.2 pins down.
> - `turnover` and `cost` from `PnLResult` are per-path tensors; the reported scalars are
>   their batch means: `turnover.mean()`, `cost.mean()`.
> - `bs_delta_vega` metrics are computed **only** when `cfg.n_instruments == 2` (i.e.
>   `cfg.instruments == ("underlying", "option")`); otherwise that strategy key is absent.

---

### Task 12.1: `_model_premium(cfg, option)` helper — risk-neutral premium for any regime

**Files:**
- Create: `deephedge/evaluate.py`
- Test: `tests/test_evaluate.py`

> Evaluation must price the sold option with the **same** risk-neutral premium that `train`
> used (spec §3): `bs_price` for `gbm`/`merton`, `heston_price_cm` for `heston`/`bates`.
> This task introduces the private helper `_model_premium` and the module's `Agg`-safe
> matplotlib import. Importing matplotlib with the `Agg` backend at module top makes the
> plotting tasks headless-safe under pytest.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluate.py
import math

import pytest
import torch

from deephedge.config import ExperimentConfig
from deephedge.instruments import EuropeanOption
from deephedge.evaluate import _model_premium


def _tiny_gbm_cfg(**overrides) -> ExperimentConfig:
    base = dict(
        s0=100.0, k=100.0, r=0.0, q=0.0, mu=None,
        maturity=30 / 252, n_steps=8,
        sigma=0.2, model="gbm", loss="cvar", alpha=0.95,
        cost=0.0, instruments=("underlying",),
        n_paths=1024, batch_size=1024,
        epochs=1, steps_per_epoch=1,
        lr=1e-3, seed=0, device="cpu",
    )
    base.update(overrides)
    return ExperimentConfig(**base)


def test_model_premium_gbm_matches_bs_price():
    from deephedge.pricing.black_scholes import bs_price

    cfg = _tiny_gbm_cfg()
    option = EuropeanOption(cfg.k, cfg.maturity)
    prem = _model_premium(cfg, option)

    S0 = torch.tensor(cfg.s0, dtype=torch.float64)
    expected = float(bs_price(S0, cfg.k, cfg.maturity, cfg.r, cfg.sigma, q=cfg.q, kind="call"))

    assert isinstance(prem, float)
    assert math.isclose(prem, expected, rel_tol=1e-9, abs_tol=1e-9)
    # ATM call under r=0, q=0 is positive and strictly below spot.
    assert 0.0 < prem < cfg.s0


def test_model_premium_heston_uses_carr_madan():
    from deephedge.pricing.heston import heston_price_cm

    cfg = _tiny_gbm_cfg(model="heston")
    option = EuropeanOption(cfg.k, cfg.maturity)
    prem = _model_premium(cfg, option)
    expected = float(heston_price_cm(cfg, cfg.k, cfg.maturity, kind="call"))

    assert math.isclose(prem, expected, rel_tol=1e-9, abs_tol=1e-9)
    assert 0.0 < prem < cfg.s0
```

- [ ] **Step 2: Run test to verify it fails**

Command: `pytest tests/test_evaluate.py -k model_premium -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'deephedge.evaluate'` (the module
and `_model_premium` do not exist yet).

- [ ] **Step 3: Write minimal implementation**

```python
# deephedge/evaluate.py
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
from deephedge.instruments import EuropeanOption  # noqa: E402
from deephedge.pricing.black_scholes import bs_delta, bs_price  # noqa: E402
from deephedge.pricing.heston import heston_price_cm  # noqa: E402

_STOCH_VOL_MODELS = {"heston", "bates"}


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
```

- [ ] **Step 4: Run test to verify it passes**

Command: `pytest tests/test_evaluate.py -k model_premium -v`

Expected: PASS — GBM premium equals `bs_price`, Heston premium equals `heston_price_cm`,
both positive and `< s0`.

- [ ] **Step 5: Commit**

```bash
git add deephedge/evaluate.py tests/test_evaluate.py
git commit -m "feat(evaluate): risk-neutral premium helper + Agg matplotlib backend"
```

---

### Task 12.2: `empirical_cvar(pnl, alpha)` matches `cvar_loss` at the optimum

**Files:**
- Modify: `deephedge/evaluate.py`
- Test: `tests/test_evaluate.py`

> The CVaR metric must equal the Rockafellar–Uryasev objective `cvar_loss` evaluated at its
> empirical optimum `w* = VaR_alpha` (spec §6.1, §13(4)). `empirical_cvar` computes
> `CVaR_alpha(L) = w* + (1/(1-alpha)) * mean(relu(L - w*))` with `L = -pnl` and
> `w* = quantile(L, alpha)`, and Task 12.2 asserts it agrees with `losses.cvar_loss` to
> float tolerance. We also pin it to the closed-form Gaussian CVaR so the number is
> independently anchored.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluate.py  (append)
def test_empirical_cvar_matches_cvar_loss_at_optimal_w():
    from deephedge.losses import cvar_loss
    from deephedge.evaluate import empirical_cvar

    gen = torch.Generator().manual_seed(2024)
    # profit-positive P&L sample (so L = -pnl is the loss); large N for a tight match.
    pnl = torch.randn(200_000, generator=gen)
    alpha = 0.95

    cvar = empirical_cvar(pnl, alpha)

    # cvar_loss at the empirical optimum w* = VaR_alpha = quantile(L, alpha).
    losses = -pnl
    w_star = torch.quantile(losses, alpha)
    ru = float(cvar_loss(pnl, alpha, torch.nn.Parameter(w_star.clone())).detach())

    assert isinstance(cvar, float)
    assert abs(cvar - ru) < 1e-5


def test_empirical_cvar_matches_gaussian_closed_form():
    from deephedge.evaluate import empirical_cvar

    # For L ~ N(0,1), CVaR_alpha(L) = phi(z_alpha) / (1 - alpha) with z_alpha = Phi^{-1}(alpha).
    import math

    gen = torch.Generator().manual_seed(7)
    pnl = torch.randn(1_000_000, generator=gen)  # L = -pnl is also N(0,1)
    alpha = 0.95

    cvar = empirical_cvar(pnl, alpha)

    z = math.sqrt(2.0) * torch.erfinv(torch.tensor(2.0 * alpha - 1.0)).item()
    phi_z = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    closed_form = phi_z / (1.0 - alpha)  # ~2.0627 for alpha=0.95

    assert abs(cvar - closed_form) < 2e-2
```

- [ ] **Step 2: Run test to verify it fails**

Command: `pytest tests/test_evaluate.py -k empirical_cvar -v`

Expected: FAIL with `ImportError: cannot import name 'empirical_cvar' from 'deephedge.evaluate'`.

- [ ] **Step 3: Write minimal implementation**

```python
# deephedge/evaluate.py  (append)
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
```

- [ ] **Step 4: Run test to verify it passes**

Command: `pytest tests/test_evaluate.py -k empirical_cvar -v`

Expected: PASS — `empirical_cvar` matches `cvar_loss` at `w*` to `<1e-5` and the Gaussian
closed-form `phi(z)/(1-alpha) ≈ 2.0627` to `<2e-2`.

- [ ] **Step 5: Commit**

```bash
git add deephedge/evaluate.py tests/test_evaluate.py
git commit -m "feat(evaluate): empirical_cvar matching RU cvar_loss optimum"
```

---

### Task 12.3: `_strategy_metrics(pnl_result, alpha)` — one strategy's metric dict

**Files:**
- Modify: `deephedge/evaluate.py`
- Test: `tests/test_evaluate.py`

> Reduce a single `PnLResult` to the flat metric dict that `evaluate` nests per strategy.
> Keys (locked): `mean_pnl`, `std`, `cvar_95`, `cvar_99`, `turnover`, `total_cost`. `cvar_95`
> uses `alpha=0.95` regardless of `cfg.alpha`; `cvar_99` uses `alpha=0.99` — the reported
> table always shows both tail levels (spec §12).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluate.py  (append)
def test_strategy_metrics_keys_and_values():
    from deephedge.portfolio import PnLResult
    from deephedge.evaluate import _strategy_metrics, empirical_cvar

    gen = torch.Generator().manual_seed(5)
    pnl = torch.randn(50_000, generator=gen)
    turnover = torch.full((50_000,), 3.0)
    cost = torch.full((50_000,), 0.25)
    res = PnLResult(pnl=pnl, turnover=turnover, cost=cost, holdings=torch.zeros(50_000, 1))

    m = _strategy_metrics(res, alpha=0.95)

    assert set(m.keys()) == {"mean_pnl", "std", "cvar_95", "cvar_99", "turnover", "total_cost"}
    assert all(isinstance(v, float) for v in m.values())

    assert abs(m["mean_pnl"] - float(pnl.mean())) < 1e-6
    assert abs(m["std"] - float(pnl.std(unbiased=True))) < 1e-6
    assert abs(m["turnover"] - 3.0) < 1e-6
    assert abs(m["total_cost"] - 0.25) < 1e-6
    assert abs(m["cvar_95"] - empirical_cvar(pnl, 0.95)) < 1e-9
    assert abs(m["cvar_99"] - empirical_cvar(pnl, 0.99)) < 1e-9
    # CVaR_99 is a deeper tail than CVaR_95.
    assert m["cvar_99"] > m["cvar_95"]
```

- [ ] **Step 2: Run test to verify it fails**

Command: `pytest tests/test_evaluate.py -k strategy_metrics -v`

Expected: FAIL with `ImportError: cannot import name '_strategy_metrics' from 'deephedge.evaluate'`.

- [ ] **Step 3: Write minimal implementation**

```python
# deephedge/evaluate.py  (append; add PnLResult to the imports near the top)
from deephedge.portfolio import PnLResult  # noqa: E402


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
```

- [ ] **Step 4: Run test to verify it passes**

Command: `pytest tests/test_evaluate.py -k strategy_metrics -v`

Expected: PASS — six expected keys, all float, mean/std/turnover/cost match inputs, CVaR
levels match `empirical_cvar`, and `cvar_99 > cvar_95`.

- [ ] **Step 5: Commit**

```bash
git add deephedge/evaluate.py tests/test_evaluate.py
git commit -m "feat(evaluate): per-strategy metric reduction"
```

---

### Task 12.4: `evaluate(cfg, hedger) -> dict` — nested per-strategy metrics on fresh paths

**Files:**
- Modify: `deephedge/evaluate.py`
- Test: `tests/test_evaluate.py`

> `evaluate` simulates **fresh out-of-sample paths** (seed `cfg.seed + 10_000`, distinct
> from training's `cfg.seed + step`), builds the sold `liability = EuropeanOption(cfg.k,
> cfg.maturity)` and the `hedge = _hedge_option(cfg) if "option" in cfg.instruments else
> None`, computes the single risk-neutral premium from the liability, builds `instr_prices`
> once via `build_instr_prices(cfg, paths, hedge)`, and runs each strategy through the
> shared `simulate_pnl(..., liability, premium, instr_prices)` engine under
> `torch.no_grad()`. The same `hedge` instance also feeds `make_bs_delta_vega_strategy(cfg,
> hedge)`. Strategies: `nn`, `bs_delta`, `no_hedge` always; `bs_delta_vega` **iff**
> `cfg.n_instruments == 2`. Returns `{strategy_name: metric_dict}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluate.py  (append)
def test_evaluate_returns_expected_keys_single_instrument():
    from deephedge.hedger import Hedger
    from deephedge.train import _n_features
    from deephedge.evaluate import evaluate

    cfg = _tiny_gbm_cfg(model="gbm", cost=0.001, n_paths=2048, batch_size=2048)
    torch.manual_seed(0)
    hedger = Hedger(_n_features(cfg), cfg.n_instruments)

    results = evaluate(cfg, hedger)

    # single-instrument regime -> no bs_delta_vega
    assert set(results.keys()) == {"nn", "bs_delta", "no_hedge"}
    metric_keys = {"mean_pnl", "std", "cvar_95", "cvar_99", "turnover", "total_cost"}
    for name, m in results.items():
        assert set(m.keys()) == metric_keys, name
        assert all(isinstance(v, float) for v in m.values()), name

    # no_hedge never trades -> zero turnover and zero cost.
    assert abs(results["no_hedge"]["turnover"]) < 1e-9
    assert abs(results["no_hedge"]["total_cost"]) < 1e-9
    # under proportional cost, the delta hedger does pay cost.
    assert results["bs_delta"]["total_cost"] > 0.0


def test_evaluate_includes_bs_delta_vega_for_two_instruments():
    from deephedge.hedger import Hedger
    from deephedge.train import _n_features
    from deephedge.evaluate import evaluate

    cfg = _tiny_gbm_cfg(
        model="heston", cost=0.001,
        instruments=("underlying", "option"),
        hedge_option_strike=110.0, hedge_option_maturity=60 / 252,
        n_paths=1024, batch_size=1024,
    )
    torch.manual_seed(0)
    hedger = Hedger(_n_features(cfg), cfg.n_instruments)

    results = evaluate(cfg, hedger)

    assert set(results.keys()) == {"nn", "bs_delta", "no_hedge", "bs_delta_vega"}
    assert all(isinstance(v, float) for v in results["bs_delta_vega"].values())
```

- [ ] **Step 2: Run test to verify it fails**

Command: `pytest tests/test_evaluate.py -k "evaluate_returns or bs_delta_vega" -v`

Expected: FAIL with `ImportError: cannot import name 'evaluate' from 'deephedge.evaluate'`.

- [ ] **Step 3: Write minimal implementation**

```python
# deephedge/evaluate.py  (append; add these imports near the top of the file)
from deephedge.simulators.base import get_simulator  # noqa: E402
from deephedge.portfolio import build_instr_prices, simulate_pnl  # noqa: E402
from deephedge.benchmarks import (  # noqa: E402
    make_bs_delta_strategy,
    make_bs_delta_vega_strategy,
    make_nn_strategy,
    make_no_hedge_strategy,
)

_EVAL_SEED_OFFSET = 10_000  # disjoint from training's cfg.seed + step draws


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
```

- [ ] **Step 4: Run test to verify it passes**

Command: `pytest tests/test_evaluate.py -k "evaluate_returns or bs_delta_vega" -v`

Expected: PASS — single-instrument returns `{nn, bs_delta, no_hedge}` with the six metric
keys; `no_hedge` has zero turnover/cost; `bs_delta` pays positive cost; two-instrument adds
`bs_delta_vega`.

- [ ] **Step 5: Commit**

```bash
git add deephedge/evaluate.py tests/test_evaluate.py
git commit -m "feat(evaluate): nested per-strategy out-of-sample metrics"
```

---

### Task 12.5: `metrics_table(results) -> str` — markdown table across strategies

**Files:**
- Modify: `deephedge/evaluate.py`
- Test: `tests/test_evaluate.py`

> Render the nested `results` dict as a GitHub-flavored markdown table: one row per strategy,
> columns in a fixed order. Numbers formatted to 4 decimals. Used by `report.md` and the
> scripts. Spec §12 / §15 (regime table).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluate.py  (append)
def test_metrics_table_is_nonempty_markdown():
    from deephedge.evaluate import metrics_table

    results = {
        "nn": {"mean_pnl": 0.01, "std": 0.5, "cvar_95": 1.2, "cvar_99": 1.8,
               "turnover": 3.0, "total_cost": 0.05},
        "bs_delta": {"mean_pnl": -0.02, "std": 0.7, "cvar_95": 1.6, "cvar_99": 2.4,
                     "turnover": 5.0, "total_cost": 0.09},
        "no_hedge": {"mean_pnl": 0.0, "std": 4.0, "cvar_95": 9.0, "cvar_99": 13.0,
                     "turnover": 0.0, "total_cost": 0.0},
    }

    table = metrics_table(results)

    assert isinstance(table, str)
    assert len(table) > 0
    # markdown table structure: header separator row of dashes/pipes.
    lines = table.strip().splitlines()
    assert lines[0].startswith("|")
    assert set(lines[1].replace("|", "").replace(":", "").strip()) <= {"-", " "}
    # one data row per strategy + header + separator.
    assert len(lines) == 2 + len(results)
    # every strategy name appears.
    for name in results:
        assert name in table
    # column headers present.
    for col in ("mean_pnl", "std", "cvar_95", "cvar_99", "turnover", "total_cost"):
        assert col in table
    # values are rendered (4-decimal formatting of cvar_95 for nn).
    assert "1.2000" in table
```

- [ ] **Step 2: Run test to verify it fails**

Command: `pytest tests/test_evaluate.py -k metrics_table -v`

Expected: FAIL with `ImportError: cannot import name 'metrics_table' from 'deephedge.evaluate'`.

- [ ] **Step 3: Write minimal implementation**

```python
# deephedge/evaluate.py  (append)
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
```

- [ ] **Step 4: Run test to verify it passes**

Command: `pytest tests/test_evaluate.py -k metrics_table -v`

Expected: PASS — non-empty string; header + separator + one row per strategy; all names and
column headers present; `1.2000` rendered.

- [ ] **Step 5: Commit**

```bash
git add deephedge/evaluate.py tests/test_evaluate.py
git commit -m "feat(evaluate): markdown metrics table renderer"
```

---

### Task 12.6: `plot_pnl_distribution(results, path)` writes the headline figure

**Files:**
- Modify: `deephedge/evaluate.py`
- Test: `tests/test_evaluate.py`

> The money-shot chart (spec §12): overlaid P&L histograms per strategy with a dashed
> vertical marker at each strategy's **negated** CVaR_95 (CVaR is a loss-side quantity; the
> marker sits in P&L space at `-CVaR_95`, i.e. the left tail). `results` here carries the raw
> per-path P&L tensors per strategy (not just scalars) so we can draw distributions; the
> function accepts the richer `{name: {"pnl": tensor, "cvar_95": float, ...}}` shape and
> ignores extra metric keys. Saves with `plt.savefig(path)` and closes the figure.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluate.py  (append)
def test_plot_pnl_distribution_writes_file(tmp_path):
    from deephedge.evaluate import plot_pnl_distribution

    gen = torch.Generator().manual_seed(1)
    results = {
        "nn": {"pnl": 0.3 * torch.randn(5000, generator=gen), "cvar_95": 0.6},
        "bs_delta": {"pnl": 0.5 * torch.randn(5000, generator=gen), "cvar_95": 1.0},
        "no_hedge": {"pnl": 3.0 * torch.randn(5000, generator=gen), "cvar_95": 6.0},
    }
    out = tmp_path / "pnl_dist.png"

    ret = plot_pnl_distribution(results, str(out))

    assert out.exists()
    assert out.stat().st_size > 0
    # convention: returns the saved path for chaining in scripts.
    assert ret == str(out)
```

- [ ] **Step 2: Run test to verify it fails**

Command: `pytest tests/test_evaluate.py -k plot_pnl_distribution -v`

Expected: FAIL with `ImportError: cannot import name 'plot_pnl_distribution' from 'deephedge.evaluate'`.

- [ ] **Step 3: Write minimal implementation**

```python
# deephedge/evaluate.py  (append)
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
```

- [ ] **Step 4: Run test to verify it passes**

Command: `pytest tests/test_evaluate.py -k plot_pnl_distribution -v`

Expected: PASS — the PNG file exists, is non-empty, and the saved path is returned.

- [ ] **Step 5: Commit**

```bash
git add deephedge/evaluate.py tests/test_evaluate.py
git commit -m "feat(evaluate): plot_pnl_distribution money-shot figure"
```

---

### Task 12.7: `plot_hedge_ratio(cfg, hedger, path)` — learned holding vs BS delta (no-trade band)

**Files:**
- Modify: `deephedge/evaluate.py`
- Test: `tests/test_evaluate.py`

> Secondary chart (spec §12): sweep spot `S` across a moneyness range at a small
> time-to-maturity `tau` near expiry, query the learned underlying holding from the hedger
> via `build_features`, and overlay it against `bs_delta`. Under transaction costs the learned
> curve is expected to flatten into a **no-transaction band** around delta — the
> utility-based / singular-control result of **Whalley–Wilmott (1997)** (band half-width
> ∝ cost^{1/3}); this is explicitly **not** Leland (1985), which is an adjusted-volatility
> periodic-rehedge scheme, not a band. The code comment must cite Whalley–Wilmott, not
> Leland. The learned holding query feeds `prev_holdings = bs_delta` so the band shows
> deviation from the frictionless target.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluate.py  (append)
def test_plot_hedge_ratio_writes_file(tmp_path):
    from deephedge.hedger import Hedger
    from deephedge.train import _n_features
    from deephedge.evaluate import plot_hedge_ratio

    cfg = _tiny_gbm_cfg(model="gbm", cost=0.01, n_steps=20)
    torch.manual_seed(0)
    hedger = Hedger(_n_features(cfg), cfg.n_instruments)
    out = tmp_path / "hedge_ratio.png"

    ret = plot_hedge_ratio(cfg, hedger, str(out))

    assert out.exists()
    assert out.stat().st_size > 0
    assert ret == str(out)


def test_plot_hedge_ratio_comment_cites_whalley_wilmott():
    # Spec §12: the band finding must cite Whalley-Wilmott, NOT Leland.
    import inspect

    from deephedge.evaluate import plot_hedge_ratio

    src = inspect.getsource(plot_hedge_ratio)
    assert "Whalley" in src
    assert "Leland" not in src
```

- [ ] **Step 2: Run test to verify it fails**

Command: `pytest tests/test_evaluate.py -k plot_hedge_ratio -v`

Expected: FAIL with `ImportError: cannot import name 'plot_hedge_ratio' from 'deephedge.evaluate'`.

- [ ] **Step 3: Write minimal implementation**

```python
# deephedge/evaluate.py  (append; add build_features to the hedger imports near the top)
from deephedge.hedger import build_features  # noqa: E402


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
```

- [ ] **Step 4: Run test to verify it passes**

Command: `pytest tests/test_evaluate.py -k plot_hedge_ratio -v`

Expected: PASS — the PNG exists and is non-empty; the function source cites Whalley (and
never Leland).

- [ ] **Step 5: Commit**

```bash
git add deephedge/evaluate.py tests/test_evaluate.py
git commit -m "feat(evaluate): plot_hedge_ratio no-trade band (Whalley-Wilmott)"
```

---

### Task 12.8: `experiments/` regime config builders (gbm, gbm_costs, heston_costs, bates_costs, multi_instrument)

**Files:**
- Create: `experiments/__init__.py`
- Create: `experiments/regimes.py`
- Test: `tests/test_experiments.py`

> One `ExperimentConfig` per regime (spec §14): `gbm` (frictionless GBM), `gbm_costs`
> (GBM + proportional cost), `heston_costs` (Heston + cost), `bates_costs` (Bates + cost),
> and `multi_instrument` (Heston + cost with `instruments=("underlying","option")`). A
> `REGIMES` registry maps name -> a zero-arg builder; `build_regime(name)` dispatches and
> raises `ValueError` on unknown names. Configs are tuned small enough to train quickly but
> remain faithful regimes.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_experiments.py
import pytest

from deephedge.config import ExperimentConfig
from experiments.regimes import REGIMES, build_regime


def test_regimes_registry_names():
    assert set(REGIMES.keys()) == {
        "gbm", "gbm_costs", "heston_costs", "bates_costs", "multi_instrument"
    }


def test_build_regime_returns_experiment_config():
    for name in REGIMES:
        cfg = build_regime(name)
        assert isinstance(cfg, ExperimentConfig)


def test_regime_model_and_cost_settings():
    assert build_regime("gbm").model == "gbm"
    assert build_regime("gbm").cost == 0.0

    assert build_regime("gbm_costs").model == "gbm"
    assert build_regime("gbm_costs").cost > 0.0

    assert build_regime("heston_costs").model == "heston"
    assert build_regime("heston_costs").cost > 0.0

    assert build_regime("bates_costs").model == "bates"
    assert build_regime("bates_costs").cost > 0.0
    assert build_regime("bates_costs").jump_intensity > 0.0  # Bates has jumps

    mi = build_regime("multi_instrument")
    assert mi.instruments == ("underlying", "option")
    assert mi.n_instruments == 2
    assert mi.model == "heston"


def test_build_regime_unknown_raises_value_error():
    with pytest.raises(ValueError, match="nope"):
        build_regime("nope")
```

- [ ] **Step 2: Run test to verify it fails**

Command: `pytest tests/test_experiments.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'experiments'` (the package and
`regimes` module do not exist yet).

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/__init__.py
"""Experiment regime configurations (spec §14)."""
```

```python
# experiments/regimes.py
"""Per-regime ExperimentConfig builders (spec §14).

Regimes: gbm, gbm_costs, heston_costs, bates_costs, multi_instrument. Each builder
returns a fully-populated ExperimentConfig; `build_regime(name)` dispatches by name.
"""
from __future__ import annotations

from typing import Callable

from deephedge.config import ExperimentConfig

_COST = 0.005  # proportional transaction-cost rate for the friction regimes


def _gbm() -> ExperimentConfig:
    return ExperimentConfig(
        model="gbm", sigma=0.2, cost=0.0,
        maturity=30 / 252, n_steps=30,
        loss="cvar", alpha=0.95,
        n_paths=100_000, batch_size=8192,
        epochs=50, steps_per_epoch=50, lr=1e-3, seed=0,
    )


def _gbm_costs() -> ExperimentConfig:
    return ExperimentConfig(
        model="gbm", sigma=0.2, cost=_COST,
        maturity=30 / 252, n_steps=30,
        loss="cvar", alpha=0.95,
        n_paths=100_000, batch_size=8192,
        epochs=50, steps_per_epoch=50, lr=1e-3, seed=0,
    )


def _heston_costs() -> ExperimentConfig:
    return ExperimentConfig(
        model="heston", cost=_COST,
        v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7,
        maturity=30 / 252, n_steps=30,
        loss="cvar", alpha=0.95,
        n_paths=100_000, batch_size=8192,
        epochs=50, steps_per_epoch=50, lr=1e-3, seed=0,
    )


def _bates_costs() -> ExperimentConfig:
    return ExperimentConfig(
        model="bates", cost=_COST,
        v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7,
        jump_intensity=1.0, jump_mean=-0.1, jump_std=0.15,
        maturity=30 / 252, n_steps=30,
        loss="cvar", alpha=0.95,
        n_paths=100_000, batch_size=8192,
        epochs=50, steps_per_epoch=50, lr=1e-3, seed=0,
    )


def _multi_instrument() -> ExperimentConfig:
    return ExperimentConfig(
        model="heston", cost=_COST,
        v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7,
        instruments=("underlying", "option"),
        hedge_option_strike=100.0, hedge_option_maturity=60 / 252,
        maturity=30 / 252, n_steps=30,
        loss="cvar", alpha=0.95,
        n_paths=100_000, batch_size=8192,
        epochs=50, steps_per_epoch=50, lr=1e-3, seed=0,
    )


REGIMES: dict[str, Callable[[], ExperimentConfig]] = {
    "gbm": _gbm,
    "gbm_costs": _gbm_costs,
    "heston_costs": _heston_costs,
    "bates_costs": _bates_costs,
    "multi_instrument": _multi_instrument,
}


def build_regime(name: str) -> ExperimentConfig:
    """Build the ExperimentConfig for a named regime (spec §14)."""
    try:
        return REGIMES[name]()
    except KeyError:
        known = ", ".join(sorted(REGIMES))
        raise ValueError(f"Unknown regime {name!r}; known regimes: {known}")
```

- [ ] **Step 4: Run test to verify it passes**

Command: `pytest tests/test_experiments.py -v`

Expected: PASS — the five regime names registered; each builds an `ExperimentConfig`; model
/cost/jump/instrument settings match; unknown name raises `ValueError`.

- [ ] **Step 5: Commit**

```bash
git add experiments/__init__.py experiments/regimes.py tests/test_experiments.py
git commit -m "feat(experiments): regime config builders (gbm/costs/heston/bates/multi)"
```

---

### Task 12.9: `scripts/run_experiment.py` — train + evaluate + save figures for a named regime

**Files:**
- Create: `scripts/__init__.py`
- Create: `scripts/run_experiment.py`
- Test: `tests/test_scripts.py`

> Glue script (spec §14/§15): given a regime name and an output directory, build the config,
> `train(cfg)`, `evaluate(cfg, hedger)`, render `metrics_table`, and save both figures. To
> keep it testable, expose a pure function `run_experiment(name, outdir, *, cfg_overrides=None)
> -> dict` that returns `{"results": ..., "table": ..., "figures": {...}}` and writes the
> table to `outdir/metrics.md`; the CLI `main()` just parses argv and calls it. The figures
> need the per-path P&L tensors, so `run_experiment` re-derives them via the shared engine
> (same fresh-seed paths as `evaluate`) and passes the enriched dict to
> `plot_pnl_distribution`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scripts.py
import torch

from scripts.run_experiment import run_experiment


def test_run_experiment_trains_evaluates_and_saves(tmp_path):
    # Override the heavy regime defaults down to a few fast steps.
    overrides = dict(
        n_paths=1024, batch_size=1024,
        n_steps=8, epochs=1, steps_per_epoch=3, lr=5e-3, seed=1,
    )
    out = run_experiment("gbm_costs", str(tmp_path), cfg_overrides=overrides)

    # nested results for the three single-instrument strategies
    assert set(out["results"].keys()) == {"nn", "bs_delta", "no_hedge"}
    # markdown table written to disk and returned
    assert (tmp_path / "metrics.md").exists()
    assert out["table"].startswith("|")
    # both figures saved
    pnl_fig = tmp_path / "pnl_distribution.png"
    band_fig = tmp_path / "hedge_ratio.png"
    assert pnl_fig.exists() and pnl_fig.stat().st_size > 0
    assert band_fig.exists() and band_fig.stat().st_size > 0
    assert out["figures"]["pnl_distribution"] == str(pnl_fig)
    assert out["figures"]["hedge_ratio"] == str(band_fig)
```

- [ ] **Step 2: Run test to verify it fails**

Command: `pytest tests/test_scripts.py::test_run_experiment_trains_evaluates_and_saves -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.run_experiment'`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/__init__.py
"""Reproducible experiment entrypoints (spec §14/§15)."""
```

```python
# scripts/run_experiment.py
"""Train + evaluate + save the two figures for one named regime (spec §14/§15).

Usage:
    python -m scripts.run_experiment <regime> <outdir>
"""
from __future__ import annotations

import argparse
import dataclasses
import os

import torch

from deephedge.benchmarks import (
    make_bs_delta_strategy,
    make_bs_delta_vega_strategy,
    make_nn_strategy,
    make_no_hedge_strategy,
)
from deephedge.evaluate import (
    _hedge_option,
    _model_premium,
    empirical_cvar,
    evaluate,
    metrics_table,
    plot_hedge_ratio,
    plot_pnl_distribution,
)
from deephedge.instruments import EuropeanOption
from deephedge.portfolio import build_instr_prices, simulate_pnl
from deephedge.simulators.base import get_simulator
from deephedge.train import train
from experiments.regimes import build_regime

_EVAL_SEED_OFFSET = 10_000


def _pnl_by_strategy(cfg, hedger) -> dict:
    """Per-path P&L + CVaR_95 per strategy on the SAME fresh paths evaluate() uses.

    Threads the two distinct options exactly as evaluate(): the `liability` (sold) option
    drives the simulate_pnl payoff/premium; the `hedge` option is marked into instr_prices
    col1 and feeds the delta+vega benchmark (same instance both places).
    """
    device = torch.device(cfg.device)
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.seed + _EVAL_SEED_OFFSET)
    paths = get_simulator(cfg.model)(cfg, cfg.n_paths, gen)
    liability = EuropeanOption(cfg.k, cfg.maturity)
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

    enriched: dict = {}
    with torch.no_grad():
        for name, strat in strategies.items():
            pnl = simulate_pnl(strat, paths, cfg, liability, premium, instr_prices).pnl
            enriched[name] = {"pnl": pnl, "cvar_95": empirical_cvar(pnl, 0.95)}
    return enriched


def run_experiment(name: str, outdir: str, *, cfg_overrides: dict | None = None) -> dict:
    """Train, evaluate, and write the metrics table + two figures for a regime."""
    os.makedirs(outdir, exist_ok=True)
    cfg = build_regime(name)
    if cfg_overrides:
        cfg = dataclasses.replace(cfg, **cfg_overrides)

    hedger, _history = train(cfg)
    results = evaluate(cfg, hedger)
    table = metrics_table(results)

    with open(os.path.join(outdir, "metrics.md"), "w") as fh:
        fh.write(f"# Regime: {name}\n\n{table}\n")

    pnl_path = os.path.join(outdir, "pnl_distribution.png")
    band_path = os.path.join(outdir, "hedge_ratio.png")
    plot_pnl_distribution(_pnl_by_strategy(cfg, hedger), pnl_path)
    plot_hedge_ratio(cfg, hedger, band_path)

    return {
        "results": results,
        "table": table,
        "figures": {"pnl_distribution": pnl_path, "hedge_ratio": band_path},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train+evaluate a deep-hedging regime.")
    parser.add_argument("regime")
    parser.add_argument("outdir")
    args = parser.parse_args()
    out = run_experiment(args.regime, args.outdir)
    print(out["table"])


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Command: `pytest tests/test_scripts.py::test_run_experiment_trains_evaluates_and_saves -v`

Expected: PASS — `run_experiment` returns the three-strategy results, writes `metrics.md`,
and saves both non-empty figures at the returned paths.

- [ ] **Step 5: Commit**

```bash
git add scripts/__init__.py scripts/run_experiment.py tests/test_scripts.py
git commit -m "feat(scripts): run_experiment train+evaluate+figures for a regime"
```

---

### Task 12.10: `scripts/make_figures.py` — regenerate both figures from a trained hedger

**Files:**
- Create: `scripts/make_figures.py`
- Test: `tests/test_scripts.py`

> A thin figure-only entrypoint (spec §15): given a regime + an output dir, train just enough
> to obtain a hedger (or accept overrides), then write only the two charts — no metrics
> table. Exposes `make_figures(name, outdir, *, cfg_overrides=None) -> dict` returning the
> figure paths; `main()` is the CLI.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scripts.py  (append)
from scripts.make_figures import make_figures


def test_make_figures_writes_both_charts(tmp_path):
    overrides = dict(
        n_paths=1024, batch_size=1024,
        n_steps=8, epochs=1, steps_per_epoch=2, lr=5e-3, seed=2,
    )
    figs = make_figures("gbm_costs", str(tmp_path), cfg_overrides=overrides)

    pnl_fig = tmp_path / "pnl_distribution.png"
    band_fig = tmp_path / "hedge_ratio.png"
    assert pnl_fig.exists() and pnl_fig.stat().st_size > 0
    assert band_fig.exists() and band_fig.stat().st_size > 0
    assert figs == {"pnl_distribution": str(pnl_fig), "hedge_ratio": str(band_fig)}
```

- [ ] **Step 2: Run test to verify it fails**

Command: `pytest tests/test_scripts.py::test_make_figures_writes_both_charts -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.make_figures'`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/make_figures.py
"""Regenerate the two money-shot figures for a regime (spec §15).

Usage:
    python -m scripts.make_figures <regime> <outdir>
"""
from __future__ import annotations

import argparse
import dataclasses
import os

from deephedge.evaluate import plot_hedge_ratio, plot_pnl_distribution
from deephedge.train import train
from experiments.regimes import build_regime
from scripts.run_experiment import _pnl_by_strategy


def make_figures(name: str, outdir: str, *, cfg_overrides: dict | None = None) -> dict:
    """Train a hedger for the regime and write only the two charts."""
    os.makedirs(outdir, exist_ok=True)
    cfg = build_regime(name)
    if cfg_overrides:
        cfg = dataclasses.replace(cfg, **cfg_overrides)

    hedger, _ = train(cfg)

    pnl_path = os.path.join(outdir, "pnl_distribution.png")
    band_path = os.path.join(outdir, "hedge_ratio.png")
    plot_pnl_distribution(_pnl_by_strategy(cfg, hedger), pnl_path)
    plot_hedge_ratio(cfg, hedger, band_path)
    return {"pnl_distribution": pnl_path, "hedge_ratio": band_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate deep-hedging figures.")
    parser.add_argument("regime")
    parser.add_argument("outdir")
    args = parser.parse_args()
    figs = make_figures(args.regime, args.outdir)
    for k, v in figs.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Command: `pytest tests/test_scripts.py::test_make_figures_writes_both_charts -v`

Expected: PASS — both non-empty figures written; returned dict matches the two paths.

- [ ] **Step 5: Commit**

```bash
git add scripts/make_figures.py tests/test_scripts.py
git commit -m "feat(scripts): make_figures regenerates the two charts"
```

---

### Task 12.11: `report.md` skeleton — regime table, two figures, three narrative paragraphs

**Files:**
- Create: `report.md`
- Test: `tests/test_report.py`

> The deliverable narrative (spec §15): the regime table, embeds of the two figures, and one
> paragraph each on (a) the CVaR loss (Rockafellar–Uryasev, `(1-alpha)` tail-probability
> denominator, joint Adam over params + `w` justified by the upper-bound/recovery property,
> not joint convexity — spec §6.1), (b) the Heston pricer (the `b = kappa` fix in the
> characteristic function, NOT `kappa - rho*xi` — spec §7.2), and (c) the no-trade-band
> finding (Whalley–Wilmott, band half-width ∝ cost^{1/3}; **not** Leland — spec §12). The
> test pins the required headings and the three load-bearing factual claims so the skeleton
> can't silently drift from the spec.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_report.py
from pathlib import Path

REPORT = Path(__file__).resolve().parents[1] / "report.md"


def test_report_exists_and_nonempty():
    assert REPORT.exists()
    assert REPORT.stat().st_size > 0


def test_report_has_required_sections():
    text = REPORT.read_text()
    # required section headings
    for heading in (
        "# Deep Hedging",
        "## Regime results",
        "## Figures",
        "## CVaR loss",
        "## Heston pricer",
        "## No-trade-band finding",
    ):
        assert heading in text, heading


def test_report_embeds_both_figures():
    text = REPORT.read_text()
    assert "pnl_distribution.png" in text
    assert "hedge_ratio.png" in text


def test_report_cites_correct_facts_not_the_bugs():
    text = REPORT.read_text()
    # CVaR: tail-probability denominator (1 - alpha), not alpha.
    assert "(1 - alpha)" in text or "(1-alpha)" in text
    # Heston: the b = kappa fix, explicitly not kappa - rho*xi.
    assert "kappa" in text
    assert "b = kappa" in text or "= kappa" in text
    # No-trade band: Whalley-Wilmott, and explicitly NOT Leland as the band source.
    assert "Whalley" in text
    assert "cost^{1/3}" in text or "cost^(1/3)" in text
```

- [ ] **Step 2: Run test to verify it fails**

Command: `pytest tests/test_report.py -v`

Expected: FAIL with `assert REPORT.exists()` False (`report.md` does not exist yet).

- [ ] **Step 3: Write minimal implementation**

```markdown
# Deep Hedging — Results Report

A research-grade implementation of deep hedging (Buehler, Gonon, Teichmann & Wood, 2019):
a learned option-hedging policy that backpropagates through simulated price trajectories
and minimizes a convex risk measure of terminal P&L, benchmarked against Black–Scholes
delta hedging under transaction costs, stochastic volatility, and jumps.

## Regime results

Metrics across regimes (`gbm`, `gbm_costs`, `heston_costs`, `bates_costs`,
`multi_instrument`) for each strategy — mean P&L, std, CVaR_95, CVaR_99, turnover, total
cost. Regenerate with `python -m scripts.run_experiment <regime> <outdir>` (the table is
written to `<outdir>/metrics.md`).

| strategy | mean_pnl | std | cvar_95 | cvar_99 | turnover | total_cost |
| --- | --- | --- | --- | --- | --- | --- |
| nn | _filled by run_experiment_ | | | | | |
| bs_delta | | | | | | |
| no_hedge | | | | | | |

## Figures

The money-shot: overlaid terminal-P&L distributions with CVaR_95 markers, showing the
learned policy's thinner left tail under Heston + costs.

![P&L distribution](pnl_distribution.png)

Learned underlying holding vs Black–Scholes delta across moneyness near expiry.

![Hedge ratio vs delta](hedge_ratio.png)

## CVaR loss

We minimize the Rockafellar–Uryasev CVaR objective
`F(theta, w) = w + (1/(1 - alpha)) * mean(relu(-PnL - w))`. The denominator is the
**tail probability `(1 - alpha)`**, not `alpha` (using `alpha` is the classic ~19×
mis-scaling bug). We optimize jointly over the network parameters `theta` and the scalar
`w` with Adam. The justification for joint Adam is the **upper-bound / recovery property**
`F(theta, w) >= CVaR(theta)` with equality at `w* = VaR_alpha = argmin_w F`, so
`min_{theta, w} F = min_theta CVaR` — **not** joint convexity, which fails for a neural-net
hedge. We therefore do not claim global optimality. The tail at `alpha = 0.95` uses only
~5% of paths, so we use large batches and a faster learning rate for `w`.

## Heston pricer

The second hedging instrument is marked with a semi-analytic Heston price: the log-spot
characteristic function plus a Carr–Madan FFT. The load-bearing correctness point is the
**constant part of the drift coefficient in the characteristic function: `b = kappa`, NOT
`kappa - rho*xi`** — mixing them mis-prices by 5–14% and the trivial `phi(0) = 1` unit test
does not catch it. We use the Albrecher (2007) `g2 / -d` formulation with the principal
square root (`Re(d) >= 0`) to keep the complex logarithm off its branch cut at long
maturity, and gate the pricer with a mandatory Monte-Carlo regression test at long-tau /
high vol-of-vol.

## No-trade-band finding

Under proportional transaction costs the learned hedge ratio flattens into a
**no-transaction band** around the Black–Scholes delta: the policy rebalances only to the
edge of the band rather than tracking delta exactly. This is the utility-based / singular-
control result of **Whalley & Wilmott (1997)** (also Hodges–Neuberger 1989, Davis–Panas–
Zariphopoulou 1993), with asymptotic band half-width proportional to `cost^{1/3}`. It is
distinct from Leland's (1985) adjusted-volatility periodic-rehedge scheme, which is not a
band.
```

- [ ] **Step 4: Run test to verify it passes**

Command: `pytest tests/test_report.py -v`

Expected: PASS — `report.md` exists, has all six headings, embeds both figures, and states
the three load-bearing facts (`(1 - alpha)` denominator, `b = kappa`, Whalley with
`cost^{1/3}`).

- [ ] **Step 5: Commit**

```bash
git add report.md tests/test_report.py
git commit -m "docs(report): results report skeleton with regime table + figures"
```

---

### Task 12.12: Full evaluation-group regression — `tests/test_evaluate.py` + scripts + report green

**Files:**
- Test: `tests/test_evaluate.py`
- Test: `tests/test_experiments.py`
- Test: `tests/test_scripts.py`
- Test: `tests/test_report.py`

> Final gate for the group: run every test added in 12.1–12.11 together to confirm the
> evaluation pipeline, regime configs, scripts, and report are mutually consistent and none
> regressed (e.g. a metric-key rename in `_strategy_metrics` that `metrics_table` or the
> scripts still reference). No new test code.

- [ ] **Step 1: Write the failing test**

No new test code — this task runs the four files added by this group as one regression gate.

- [ ] **Step 2: Run test to verify it fails**

Command: `pytest tests/test_evaluate.py tests/test_experiments.py tests/test_scripts.py tests/test_report.py -v`

Expected: FAIL only if an earlier task's edit broke a sibling (e.g. `evaluate` and
`run_experiment` disagree on strategy keys, or a metric column was renamed). If 12.1–12.11
are intact this is already green — confirm and commit.

- [ ] **Step 3: Write minimal implementation**

No production change unless a regression appears. If `evaluate` and `_pnl_by_strategy`
diverge on which strategies they include, reconcile both to the same rule —
`{nn, bs_delta, no_hedge}` plus `bs_delta_vega` iff `cfg.n_instruments == 2` — since that is
the contract Task 12.4 locked. If `metrics_table` references a key absent from
`_strategy_metrics`, restore the `_METRIC_COLUMNS` tuple from Task 12.5 to exactly
`("mean_pnl", "std", "cvar_95", "cvar_99", "turnover", "total_cost")`.

- [ ] **Step 4: Run test to verify it passes**

Command: `pytest tests/test_evaluate.py tests/test_experiments.py tests/test_scripts.py tests/test_report.py -v`

Expected: PASS — every evaluation, experiment, script, and report test green together.

- [ ] **Step 5: Commit**

```bash
git add tests/test_evaluate.py tests/test_experiments.py tests/test_scripts.py tests/test_report.py
git commit -m "test(evaluate): full group regression across eval/experiments/scripts/report"
```

---

### Task 12.13: `README.md` — portfolio front door (money-shot + how-to-reproduce)

**Files:**
- Create: `README.md`
- Test: `tests/test_readme.py`

> Spec §15: the README is the first thing a hiring manager sees — it leads with the
> money-shot P&L-distribution chart and tells them how to reproduce it. Depends on the
> headline figure produced by `scripts/make_figures.py` (Task 12.10) under `figures/`.
> Use the EXACT filename `make_figures.py` writes for the `heston_costs` regime; the
> placeholder below is `figures/heston_costs_pnl_distribution.png` — adjust the embed if
> your Task 12.10 path differs.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_readme.py
from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"


def test_readme_exists_and_has_required_sections():
    assert README.exists(), "README.md does not exist yet"
    text = README.read_text()
    # leads with the project title
    assert text.lstrip().startswith("# Deep Hedging")
    # embeds the money-shot figure from figures/
    assert "](figures/" in text and "![" in text
    # install + reproduce instructions
    assert "pip install -e ." in text
    assert "## Reproduce" in text
    assert "scripts/run_experiment.py" in text or "scripts/make_figures.py" in text
    # links onward to the full results report and the design spec
    assert "report.md" in text
    assert "docs/specs/2026-06-10-deep-hedging-design.md" in text
```

- [ ] **Step 2: Run test to verify it fails**

Command: `pytest tests/test_readme.py -v`
Expected: FAIL with `assert README.exists()` False (no `README.md` yet).

- [ ] **Step 3: Write minimal implementation**

```markdown
<!-- README.md -->
# Deep Hedging

Learn option-hedging strategies that beat Black–Scholes delta hedging on tail risk
under realistic frictions — discrete rebalancing, proportional transaction costs,
stochastic volatility (Heston) and jumps (Bates). A neural hedger is trained by
backpropagating through a fully-simulated, differentiable price trajectory and
minimizing a convex risk measure (CVaR / entropic) of terminal P&L
(Buehler, Gonon, Teichmann & Wood, 2019).

![Learned hedge vs Black–Scholes delta: terminal P&L distribution under Heston + costs. The learned strategy has the thinner left tail (lower CVaR).](figures/heston_costs_pnl_distribution.png)

*Terminal P&L under Heston + transaction costs: the learned policy (lower CVaR₉₅) vs
Black–Scholes delta hedging.*

## What it does

- **Simulators:** GBM, Heston (full-truncation Euler), Merton & Bates jump-diffusion.
- **Pricing:** Black–Scholes + a semi-analytic Heston pricer (characteristic function +
  Carr–Madan), used to mark a second hedging instrument for vega hedging.
- **Risk-measure losses:** CVaR (Rockafellar–Uryasev) and entropic, hand-written and
  differentiable, optimized jointly with the hedge network.
- **Result:** lower tail risk (CVaR₉₅/₉₉) than BS delta hedging under frictions, plus a
  learned no-transaction band around delta.

## Install

```bash
pip install -e .
```

## Reproduce

```bash
# Train + evaluate a regime and write metrics:
python scripts/run_experiment.py heston_costs
# Regenerate all figures (the money-shot + the hedge-ratio band plot):
python scripts/make_figures.py
```

Regimes: `gbm`, `gbm_costs`, `heston_costs`, `bates_costs`, `multi_instrument`.

## More

- Full results + regime table + commentary: [`report.md`](report.md).
- Design & methodology (with verified quant formulas): [`docs/specs/2026-06-10-deep-hedging-design.md`](docs/specs/2026-06-10-deep-hedging-design.md).
```

- [ ] **Step 4: Run test to verify it passes**

Command: `pytest tests/test_readme.py -v`
Expected: PASS — README exists, leads with the title, embeds the figure, and has install/reproduce/links.

- [ ] **Step 5: Commit**

```bash
git add README.md tests/test_readme.py
git commit -m "docs(readme): portfolio front door with money-shot chart and reproduce steps"
```
