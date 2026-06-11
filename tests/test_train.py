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
