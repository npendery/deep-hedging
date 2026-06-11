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
