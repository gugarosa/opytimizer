import pytest

from opytimizer.core import optimizer


def test_optimizer_creation():
    new_optimizer = optimizer.Optimizer(algorithm='PSO', hyperparams_path='opytimizer/core/test.json')
    assert new_optimizer.hyperparams
