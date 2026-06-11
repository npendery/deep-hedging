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
