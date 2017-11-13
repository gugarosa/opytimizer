import pytest

from opytimizer.optimizers import pso

def test_pso_creation():
    hyperparams = {"w_min": 1.0, "w_max": 3.0}
    new_pso = pso.PSO(hyperparams=hyperparams)
    assert new_pso.hyperparams