"""Tests for deephedge.hedger: build_features and Hedger."""

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
    assert math.isclose(feats[0, 3].item(), -0.1, rel_tol=1e-5)
    assert math.isclose(feats[1, 3].item(), 0.2, rel_tol=1e-6)
    # column 4: sqrt(clamp(V_i, 0)) -> sqrt(0.04)=0.2 ; clamped negative -> 0.0
    assert math.isclose(feats[0, 4].item(), 0.2, rel_tol=1e-6)
    assert feats[1, 4].item() == 0.0


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
