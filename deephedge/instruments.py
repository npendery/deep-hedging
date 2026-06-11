from dataclasses import dataclass

import torch


@dataclass
class EuropeanOption:
    strike: float
    maturity: float
    kind: str = "call"


def payoff(option: EuropeanOption, S_T) -> torch.Tensor:
    """Terminal European payoff: relu(S_T - K) for a call, relu(K - S_T) for a put."""
    S_T = torch.as_tensor(S_T)
    if option.kind == "call":
        return torch.relu(S_T - option.strike)
    if option.kind == "put":
        return torch.relu(option.strike - S_T)
    raise ValueError(f"Unknown option kind {option.kind!r}; expected 'call' or 'put'")
