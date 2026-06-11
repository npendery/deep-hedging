import importlib

import deephedge


def test_package_exposes_version():
    assert hasattr(deephedge, "__version__")
    assert deephedge.__version__ == "0.1.0"


def test_subpackages_importable():
    # Submodules must be importable as packages.
    sim = importlib.import_module("deephedge.simulators")
    pricing = importlib.import_module("deephedge.pricing")
    assert sim is not None
    assert pricing is not None
