# tests/test_conftest_fixtures.py
import torch

from deephedge.config import ExperimentConfig


def test_generator_fixture_is_seeded_and_reproducible(generator):
    assert isinstance(generator, torch.Generator)
    a = torch.randn(4, generator=generator)
    # Re-seed to the documented test seed and draw again -> identical sequence.
    g2 = torch.Generator()
    g2.manual_seed(1234)
    b = torch.randn(4, generator=g2)
    assert torch.allclose(a, b)


def test_small_config_fixture_is_small(small_config):
    assert isinstance(small_config, ExperimentConfig)
    assert small_config.n_paths <= 1000
    assert small_config.batch_size <= small_config.n_paths
    assert small_config.n_steps <= 10
    assert small_config.device == "cpu"
