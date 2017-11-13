import pytest

from opytimizer.core import optimizer


def test_optimizer_creation():
    hyperparams = {"optimification": "True"}
    new_optimizer = optimizer.Optimizer(hyperparams=hyperparams)
    assert new_optimizer.hyperparams
