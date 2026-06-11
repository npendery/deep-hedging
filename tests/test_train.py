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


def test_train_entropic_loss_branch_has_no_w():
    cfg = _tiny_cfg(loss="entropic", entropic_lambda=1.0)
    hedger, history = train(cfg)

    n_steps_total = cfg.epochs * cfg.steps_per_epoch
    assert len(history["loss"]) == n_steps_total
    # entropic path never creates w -> empty list
    assert history["w"] == []
    # entropic loss with profit-positive PnL is finite and not NaN
    assert all(x == x for x in history["loss"])  # NaN-check (NaN != NaN)


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
