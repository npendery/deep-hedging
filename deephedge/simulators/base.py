from dataclasses import dataclass

import torch


@dataclass
class Paths:
    S: torch.Tensor            # (n_paths, n_steps+1)
    V: torch.Tensor | None     # (n_paths, n_steps+1) for heston/bates, else None
    dt: float
    times: torch.Tensor        # (n_steps+1,)
