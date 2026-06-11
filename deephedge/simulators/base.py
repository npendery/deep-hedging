from dataclasses import dataclass

import torch


@dataclass
class Paths:
    S: torch.Tensor            # (n_paths, n_steps+1)
    V: torch.Tensor | None     # (n_paths, n_steps+1) for heston/bates, else None
    dt: float
    times: torch.Tensor        # (n_steps+1,)


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
