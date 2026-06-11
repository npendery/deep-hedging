# Deep Hedging — Design Spec

**Date:** 2026-06-10
**Status:** Draft for review
**Goal:** A research-grade, reproducible implementation of *deep hedging* (Buehler,
Gonon, Teichmann & Wood, 2019) that learns option-hedging strategies by minimizing a
convex risk measure of terminal P&L, and demonstrably beats Black–Scholes delta
hedging on tail risk under realistic frictions (transaction costs, stochastic
volatility, jumps). Built as a standalone portfolio piece for a quant-research / ML
audience.

---

## 1. Why this project

Classical Black–Scholes delta hedging assumes continuous, frictionless trading. In
reality you rebalance at discrete times, pay proportional transaction costs, and vol
is stochastic — so the textbook delta is no longer optimal. Deep hedging learns the
hedging policy directly by backpropagating through a fully-simulated, differentiable
price trajectory and minimizing a risk measure (CVaR / entropic) of the hedged
portfolio's terminal P&L. Under frictions + stochastic vol it can beat the BS
benchmark, which is the clean, defensible headline result.

**The money-shot deliverable:** one chart — P&L distribution (and CVaR) of the learned
strategy vs BS delta hedging under Heston + transaction costs — showing the network
achieving lower tail risk.

This is deliberately *not* a "predict the price with an LSTM" project. It exercises
autograd through a multi-step stochastic environment, a hand-written convex
risk-measure loss, and correct quant-finance plumbing (Heston simulation + pricing,
jump-diffusion, Greeks).

---

## 2. Scope (V1 = maximal)

In scope for V1:

- **Simulators:** GBM, Heston (full-truncation Euler), Merton jump-diffusion (GBM +
  jumps), Bates (Heston + jumps).
- **Pricing:** Black–Scholes (price + Greeks), Heston semi-analytic pricer
  (characteristic function + Carr–Madan FFT), used to mark the second hedging
  instrument.
- **Hedger:** shared-weight per-timestep MLP (semi-recurrent: prior holdings fed in).
- **Risk-measure losses:** CVaR (Rockafellar–Uryasev) and entropic, config-selectable.
- **Multi-instrument hedging:** hedge with the underlying *and* a second vanilla option
  (vega/gamma hedging under stochastic vol).
- **Benchmarks:** BS delta, BS delta+vega, no-hedge.
- **Evaluation:** P&L distribution chart + CVaR table across regimes; learned
  hedge-ratio-vs-delta visualization.

Out of scope (future): American/exotic payoffs, full Andersen-QE simulator (offered as
an alternative scheme but not required for V1), AAD Greeks, live-market calibration.

---

## 3. Problem formulation

We **sell one European call** with strike `K`, maturity `T` (ATM default `K = S_0`).
Hedge over discrete times `t_0 = 0 < t_1 < ... < t_N = T` (`N` steps).

At each `t_i` the agent chooses holdings `δ_i ∈ R^m` in `m` hedging instruments
(`m = 1` underlying-only; `m = 2` underlying + a vanilla option). Rebalancing incurs
**proportional transaction cost** `c · Σ_j p^{(j)}_i · |δ^{(j)}_i − δ^{(j)}_{i-1}|`,
where `p^{(j)}_i` is the price of instrument `j` at `t_i` (the underlying `S_i` for the
stock leg, the model-marked option price for the option leg).

**Terminal hedged P&L** (we are short the call, receive premium `p0`):

```
PnL = p0
    + Σ_{i=0}^{N-1} δ_i · (P_{i+1} − P_i)      # MTM gains across all hedging instruments
    − Σ_{i=0}^{N-1} cost_i                      # transaction costs
    − max(S_N − K, 0)                           # liability: short-call payoff
```

`P_i` is the vector of hedging-instrument prices (underlying + marked option). Both the
NN and the benchmark receive the **same premium** `p0 = model price` → apples-to-apples
comparison of terminal P&L distributions.

**Objective:** minimize a convex risk measure `ρ` of the terminal P&L. We minimize
`ρ` over the network parameters (and, for CVaR, an auxiliary scalar — see §6).

### Measure convention (important)

Default: simulate paths under **`μ = r`** (risk-neutral-style drift) so that realized
P&L reflects *pure hedging error + cost*, not directional drift. This keeps the
"lower tail risk from better hedging" story clean and unconfounded. A config flag
allows `μ ≠ r` (real-world drift) to show the net can also exploit drift — a nuance,
not the headline. Option premium `p0` and BS-benchmark deltas are always computed under
the risk-neutral measure (drift `r`).

---

## 4. Architecture / module breakdown

Each module has one purpose, a typed interface, and independent unit tests.

```
deep-hedging/
  pyproject.toml              # torch, numpy, scipy, matplotlib, pytest
  README.md                   # money-shot chart + how-to
  deephedge/
    __init__.py
    config.py                 # ExperimentConfig dataclass/pydantic; seed, regime, costs, net, loss
    simulators/
      base.py                 # Simulator protocol: simulate(n_paths, n_steps, cfg) -> Paths{S, V?}
      gbm.py
      heston.py               # full-truncation Euler (+ optional Andersen QE)
      jumps.py                # Merton (GBM+jumps), Bates (Heston+jumps)
    pricing/
      black_scholes.py        # price, delta, vega, gamma, theta; put-call parity
      heston.py               # characteristic function + Carr-Madan FFT pricer + delta
    instruments.py            # EuropeanOption contract; HedgingInstrument (underlying / option)
    hedger.py                 # shared-weight MLP policy
    losses.py                 # cvar_loss, entropic_loss
    portfolio.py              # simulate hedged P&L for a strategy (NN or analytic) incl. costs
    benchmarks.py             # bs_delta, bs_delta_vega, no_hedge strategies
    train.py                  # training loop (Adam, fresh-path resampling)
    evaluate.py               # metrics (mean, std, CVaR_95/99, turnover) + plots
  experiments/                # one config per regime (gbm, gbm_costs, heston_costs, bates_costs)
  scripts/
    run_experiment.py
    make_figures.py
  tests/
  report.md / report.ipynb    # narrative results
```

Stack: **PyTorch** (autograd through the trajectory), **numpy/scipy** (Heston FFT,
stats), **matplotlib** (figures), **pytest**. Device-agnostic
(`cuda` if available else `cpu`). Global seed control for reproducibility.

---

## 5. Simulators

Common protocol: `simulate(n_paths, n_steps, cfg) -> Paths` where `Paths` holds torch
tensors `S: (n_paths, n_steps+1)` and (for stochastic-vol models) `V: (n_paths,
n_steps+1)`. All produced on the target device, seeded.

### 5.1 GBM
Exact log-Euler: `S_{i+1} = S_i · exp((μ − ½σ²)Δt + σ√Δt · Z)`.

### 5.2 Heston — full-truncation Euler
Dynamics: `dS = μ S dt + √V S dW1`, `dV = κ(θ − V)dt + ξ√V dW2`, `corr(dW1,dW2) = ρ`.

**Full-truncation Euler (verified against Lord, Koekkoek & van Dijk 2010):**

```
V_plus      = max(V_i, 0)                                  # truncate the FUNCTION fed to coeffs
V_{i+1}     = V_i + κ(θ − V_plus)Δt + ξ √(V_plus) √Δt · Z2  # carried state V_{i+1} may go negative
log S_{i+1} = log S_i + (μ − ½ V_plus)Δt + √(V_plus) √Δt · Z1
```

Verified correctness points:
- `max(·,0)` is applied in **both** the drift mean-reversion and the diffusion, via
  `V_plus = max(V_i, 0)`. The **carried state `V_{i+1}` is NOT truncated** — it is
  allowed to be negative and carried forward. Truncating the stored state would be the
  absorption/reflection family, which is a different (worse) scheme.
- Correlate the two Gaussians via Cholesky: `Z1 = Z_a`, `Z2 = ρ Z_a + √(1−ρ²) Z_b`,
  with `Z_a, Z_b` iid `N(0,1)`.
- **Feller condition** `2κθ ≥ ξ²`: when satisfied (strict), the continuous variance
  stays strictly positive; when violated, the continuous process can *touch* zero and
  reflect (0 is attainable but not absorbing). **Independently of Feller, the Euler
  discretization can still produce negative `V` because the Gaussian increment is
  unbounded — that is the actual reason truncation is needed.**
- **Caveat for our jump/high-ξ regimes:** full truncation is the best *simple* biased
  scheme, but when Feller is violated (large vol-of-vol, common in equity calibrations)
  even full truncation develops bias. We expose an **optional Andersen (2008) QE
  scheme** behind the same protocol for accuracy-sensitive regimes; full truncation is
  the V1 default.

### 5.3 Jumps — Merton & Bates
Add compound-Poisson jumps to the log-price. Over a step `Δt`: number of jumps
`n ~ Poisson(λ_J Δt)`, jump sizes `Y_k ~ N(μ_J, σ_J²)` in log-space; add `Σ_k Y_k`.

**Risk-neutral drift compensator (verified, Merton 1976 / Bates 1996):** to keep the
discounted price a martingale, the log-price drift carries the compensator
`− λ_J (E[e^Y] − 1) = − λ_J (exp(μ_J + ½σ_J²) − 1)`.

- **Merton** = GBM + these jumps (with the compensator in the drift).
- **Bates** = Heston + these jumps.
- Cross-check the Merton simulator against Merton's closed-form price (Poisson-weighted
  sum of BS prices) in tests.

---

## 6. Risk-measure losses (`losses.py`)

This is the "I understand what I'm optimizing" centerpiece. Work in **loss**
`L = −PnL` (larger = worse) consistently everywhere.

### 6.1 CVaR (Rockafellar–Uryasev) — verified
```
cvar_loss(PnL, α) = w + (1/(1−α)) · mean( relu(L − w) ),   L = −PnL
```
minimized jointly over (network params, scalar `w`) by Adam, with `w` a learnable
parameter.

Verified facts and the nuances that survived adversarial review:
- The `(1 − α)` **tail-probability** denominator is correct (using `α` is the classic
  ~19× mis-scaling bug). `relu = (·)^+` exact.
- It is a **minimization** over `w`.
- **Justification for joint Adam is the upper-bound / recovery property**, NOT joint
  convexity: for any fixed `w`, `F(params, w) ≥ CVaR(params)` because
  `CVaR = min_w F`, so `min_{params,w} F = min_{params} CVaR`. With a neural-net hedge
  the problem is **non-convex** (RU joint convexity needs the loss convex in the
  decision variable — RU 2002 Corollary 11 / RU 2000 Theorem 2). **Do not claim global
  optimality.**
- At the optimum `w* = VaR_α` only as the **left endpoint of the argmin** (unique when
  the P&L distribution is continuous — the usual Monte-Carlo case). Do not interpret
  `w` as VaR if payoffs have atoms.
- **Implementation:** the tail at `α = 0.95` uses only ~5% of paths → high-variance
  gradients → use **large/representative batches**. `w` may use its own / faster
  optimizer (well-conditioned 1-D problem). Subgradient at the relu kink is fine for
  SGD.

### 6.2 Entropic (exponential) risk — verified
```
entropic_loss(PnL, λ) = (1/λ) · ( logsumexp(−λ · PnL) − log N )
                      = (1/λ) · log( mean( exp(−λ · PnL) ) )
```
- Sign: **PnL is profit-positive**; the `exp(−λ·PnL)` exponent penalizes losses
  exponentially. Convex, monotone, cash-invariant (not coherent). Equivalent to
  maximizing CARA (`u(x) = −e^{−λx}`) expected utility; `λ` = absolute risk aversion.
- Must use **mean**, i.e. subtract `log N` (otherwise loss values are not comparable
  across batch sizes; argmin is unaffected but reporting is wrong).
- Empirical estimator is biased `O(1/N)` (Jensen); vanishes as `N → ∞`. Standard
  max-shift logsumexp stabilization is exact.

---

## 7. Pricing (`pricing/`)

### 7.1 Black–Scholes (`black_scholes.py`)
`price`, `delta = N(d1)`, `vega`, `gamma`, `theta`. Tests: known textbook values +
put–call parity.

### 7.2 Heston semi-analytic pricer (`heston.py`)
Characteristic function of log-spot + Carr–Madan FFT. Used to mark the **second hedging
instrument** (a vanilla option) each timestep under Heston/Bates.

**Verified, bug-fixed single characteristic function** (this exact form — the
constant part of the drift coefficient is `κ`, NOT `κ − ρσ`; mixing them mis-prices
5–14%):

```
β  = κ − ρσ·i·u
d  = sqrt( β² + σ²·(u² + i·u) )                      # principal sqrt, Re(d) ≥ 0
g2 = (β − d) / (β + d)                               # Albrecher "g2 / −d" form (stable)
D  = ((β − d) / σ²) · (1 − e^{−d·τ}) / (1 − g2·e^{−d·τ})
C  = (κθ / σ²) · [ (β − d)·τ − 2·ln( (1 − g2·e^{−d·τ}) / (1 − g2) ) ]
φ(u) = exp( C + D·v0 + i·u·( ln S0 + (r − q)·τ ) )
```

Verified numerical-stability points:
- Use the **Albrecher (2007) g2 / −d formulation with the principal square root**
  (`Re(d) ≥ 0`). The instability is the complex **logarithm** wrapping across its
  branch cut as `τ` grows; the g2/−d rearrangement keeps the argument in the right
  half-plane. **Never mix** principal sqrt with the `g1 / +d` form (the most commonly
  shipped bug).
- **Carr–Madan**: damped transform with damping `α ≈ 1.5`,
  `ψ(v) = e^{−rT} φ(v − (α+1)i) / (α² + α − v² + i(2α+1)v)`,
  `c_T(k) = e^{−αk}/π · ∫ Re[e^{−ivk} ψ(v)] dv`.
- **Greeks:** Heston **delta = `e^{−qT} · P1`** (cheap, exact). "Vega" is ambiguous
  (sensitivity to `v0`, `θ`, and `σ` all differ); for the hedging use-case use
  **central-difference bumping** (relative bump ~`1e-4`) for everything except delta —
  far less error-prone than differentiating the complex integrand.

**MANDATORY regression test:** validate the Heston pricer against a **Monte-Carlo
price** (full-truncation Euler, ~5×10⁵ paths) — or the two-CF Gil–Pelaez price — at
**long maturity and high vol-of-vol**. The trivial `φ(0) = 1` unit test does **not**
catch the `b = κ` vs `b = κ − ρσ` bug (both pass). This test is the gate that does.

---

## 8. Hedger network (`hedger.py`)

Shared-weight feed-forward MLP applied at each timestep (semi-recurrent: prior holdings
are an input feature). Markovian state ⇒ a per-step MLP suffices; LSTM is a stretch
ablation only.

- **Inputs** at step `i`: `[ log(S_i / K), τ_i = (T − t_i)/T, δ_{i-1} (prior holdings),
  √V_i (Heston/Bates) ]`, normalized.
- **Output:** target holdings `δ_i ∈ R^m` (one per hedging instrument).
- Small MLP (e.g. 2–3 hidden layers, ReLU). Same weights across all timesteps.

---

## 9. Portfolio / P&L engine (`portfolio.py`)

Given a strategy (NN policy *or* an analytic benchmark) and simulated paths, roll
forward the trajectory in a Python loop over `N` steps, fully differentiable in torch:
mark all hedging instruments each step (underlying directly; option leg via the BS or
Heston pricer), accumulate MTM gains, subtract proportional transaction costs, subtract
the terminal short-call payoff, add premium `p0`. Returns per-path terminal P&L tensor
(and diagnostics: turnover, total cost). The gradient flows holdings → costs → terminal
P&L → risk loss.

This single engine is shared by training (NN) and evaluation (NN + benchmarks) so the
comparison is exact.

---

## 10. Benchmarks (`benchmarks.py`)

- **BS delta hedge:** `δ_i = N(d1)`. Under GBM uses true `σ`; under Heston/Bates uses a
  naive vol estimate (e.g. current `√V_i` or fixed implied) — this is the textbook
  strawman the NN should beat under frictions.
- **BS delta+vega hedge:** for the multi-instrument case, neutralize delta + vega with
  BS Greeks — a *stronger* baseline so the NN's win is non-trivial.
- **No-hedge:** collect premium, pay payoff. Context baseline.

---

## 11. Training (`train.py`)

- **Adam**, minibatches of Monte-Carlo paths. **Fresh path simulation each
  step/epoch** ⇒ effectively infinite data, no overfitting, clean generalization story.
- Backprop through the full `N`-step trajectory.
- Large batches for CVaR (tail-gradient variance — §6.1). Separate/faster LR for the
  CVaR `w` scalar.
- Seeded; checkpoint best model by validation risk.

---

## 12. Evaluation & the money shot (`evaluate.py`)

- Out-of-sample paths (fresh seed). Compute terminal-P&L distributions for NN vs
  BS-delta vs no-hedge.
- **Headline chart:** overlaid P&L histograms/KDE with CVaR markers, showing the NN's
  **thinner left tail** under Heston + costs.
- **Secondary chart:** learned hedge ratio vs BS delta as a function of moneyness near
  expiry — expected to show a **no-transaction band** around delta under costs
  (rebalance only to the band edge). Band behavior is the utility-based / singular-
  control result (**Whalley–Wilmott 1997**, Hodges–Neuberger 1989, Davis–Panas–
  Zariphopoulou 1993; asymptotic band half-width ∝ cost^{1/3}). *Not* Leland — Leland
  (1985) is a different, adjusted-volatility periodic-rehedge scheme, not a band.
- **Metrics table** across regimes `{GBM, GBM+costs, Heston+costs, Bates+costs}`:
  mean P&L, std, CVaR_95, CVaR_99, turnover, total cost, for each strategy.

---

## 13. Correctness tests (`tests/`) — the trust-builders

1. **Black–Scholes:** price/Greeks vs textbook values; put–call parity.
2. **Heston pricer regression (mandatory):** vs Monte-Carlo (and/or two-CF Gil–Pelaez)
   at long-τ / high-ξ; also `ξ → 0` should converge to the BS price. (Guards the
   `b = κ` fix; `φ(0)=1` is necessary but insufficient.)
3. **Simulator martingale check:** under `μ = r`, discounted `E[S_T] ≈ S_0` within MC
   CI; variance scales correctly with `Δt`.
4. **CVaR loss:** matches the closed-form Gaussian CVaR for normal P&L; sanity
   `CVaR_α ≥ VaR_α ≥ E[L]`; a `~(1−α)` fraction of paths exceed `w` at the optimum.
5. **Entropic loss:** `λ → 0⁺` → `−E[PnL]`; mean-vs-sum (`log N`) identity; CARA
   equivalence on a toy normal.
6. **Replication gate (end-to-end):** frictionless, **fine-grid** GBM + BS-delta hedge
   ⇒ hedged P&L ≈ 0 with low variance (validates the whole portfolio engine). The
   trained NN in the same limit should **recover ~BS delta and drive variance → 0**.
   **Tolerance-based, not bit-exact:** the frictionless→replication result is a
   limit/approximation (price-convergence + universal approximation), and on a discrete
   grid the market is incomplete — expect small grid- and training-dependent deviations.
   Gate = agreement within grid + SGD tolerance.
7. **Hedger:** input/output shapes; gradient flows through the trajectory to params and
   to the CVaR `w`.

---

## 14. Experiments / regimes

Config-driven (`experiments/*.yaml` or dataclasses), one per regime:
`gbm`, `gbm_costs`, `heston_costs`, `bates_costs`, plus a `multi_instrument` (option
leg) variant under Heston. Each runs train → evaluate → figures reproducibly from a
seed.

---

## 15. Deliverables

- Clean installable package (`pip install -e .`), device-agnostic, seeded.
- Test suite (the §13 gates) green.
- `README.md` leading with the money-shot chart + how-to-reproduce.
- Short `report.md` / notebook: the regime table, the two charts, and a paragraph each
  on the CVaR loss, the Heston pricer, and the no-trade-band finding.

---

## 16. Build order (even at maximal scope)

1. GBM simulator + BS pricer + portfolio engine + **replication gate** (validates the
   spine).
2. CVaR + entropic losses (+ their tests).
3. Hedger net + training loop; reproduce "recovers BS delta" on frictionless GBM, then
   add costs and beat BS.
4. Heston simulator (full-truncation) + martingale test.
5. Jumps (Merton + Bates) + Merton closed-form cross-check.
6. Heston pricer (CF + Carr–Madan) + **mandatory MC regression test**.
7. Multi-instrument option-leg hedging (vega) + delta+vega benchmark.
8. Evaluation, figures, report.

---

## 17. References (verified)

- Buehler, Gonon, Teichmann & Wood (2019), *Deep Hedging*, Quantitative Finance
  19(8):1271–1291.
- Rockafellar & Uryasev (2000), *Optimization of Conditional Value-at-Risk*, J. of Risk
  2(3):21–41; (2002), *CVaR for general loss distributions*, J. Banking & Finance
  26(7):1443–1471 (Cor. 11 / Thm 2 on joint-convexity condition).
- Föllmer & Schied, *Stochastic Finance* (4th ed.), Example 4.34 (entropic risk);
  Ben-Tal & Teboulle (2007) (OCE).
- Heston (1993); Albrecher, Mayer, Schoutens & Tistaert (2007), *The Little Heston
  Trap*; Carr & Madan (1999), *Option valuation using the FFT*; Lord & Kahl (2006).
- Lord, Koekkoek & van Dijk (2010), *A comparison of biased simulation schemes…*,
  Quant. Finance 10(2):177–194; Andersen (2008), *Simple and efficient simulation of
  the Heston model* (QE scheme).
- Merton (1976) (jump-diffusion); Bates (1996) (stochastic-vol + jumps).
- Whalley & Wilmott (1997); Hodges & Neuberger (1989); Davis, Panas & Zariphopoulou
  (1993) (no-transaction band); Leland (1985) (adjusted-vol periodic rehedge — distinct
  from the band).

---

## 18. Relationship to the `investment-bot` repo (optional follow-on)

Deep hedging is a **risk-management / pricing** engine, not a stock-picker. Transferable
pieces that could harden the existing options-trading bot, *without* deploying the
sim-trained policy to live orders: the BS/Heston **pricers + Greeks** (the bot has
none today), **CVaR-based** risk limits to replace the underlying-price trailing stop
on options, **no-trade-band** cost-aware rebalancing, and **indifference pricing**
(model price vs market premium) as a vol-richness signal for the spreads/condors the
bot already trades. The sim-to-real gap is the hard limit — backtest on real option
chains before trusting any of it live.
