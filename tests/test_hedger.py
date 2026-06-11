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
