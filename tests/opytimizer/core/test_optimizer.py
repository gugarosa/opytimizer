import pytest

from opytimizer.core import optimizer


def test_optimizer_algorithm():
    new_optimizer = optimizer.Optimizer()

    assert new_optimizer.algorithm == 'PSO'


def test_optimizer_hyperparams():
    new_optimizer = optimizer.Optimizer()

    assert new_optimizer.hyperparams == None


def test_optimizer_hyperparams_setter():
    new_optimizer = optimizer.Optimizer()

    new_optimizer.hyperparams = {
        'w': 1.5
    }

    assert new_optimizer.hyperparams['w'] == 1.5


def test_optimizer_built():
    new_optimizer = optimizer.Optimizer()

    assert new_optimizer.built == False


def test_optimizer_built_setter():
    new_optimizer = optimizer.Optimizer()

    new_optimizer.built = True

    assert new_optimizer.built == True
