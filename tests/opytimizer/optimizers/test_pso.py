import numpy as np
import pytest
from opytimizer.optimizers import pso


def test_pso_hyperparams():
    hyperparams = {
        'w': 2,
        'c1': 1.7,
        'c2': 1.7
    }

    p = pso.PSO(hyperparams=hyperparams)

    assert p.w == 2

    assert p.c1 == 1.7

    assert p.c2 == 1.7


def test_pso_hyperparams_setter():
    p = pso.PSO()

    p.w = 1
    assert p.w == 1

    p.c1 = 1.5
    assert p.c1 == 1.5

    p.c2 = 1.5
    assert p.c2 == 1.5


def test_pso_local_position():
    p = pso.PSO()

    assert p.local_position == None


def test_pso_local_position_setter():
    p = pso.PSO()

    p.local_position = np.zeros((1, 1))

    assert p.local_position.shape == (1, 1)


def test_pso_velocity():
    p = pso.PSO()

    assert p.velocity == None


def test_pso_velocity_setter():
    p = pso.PSO()

    p.velocity = np.zeros((1, 1))

    assert p.velocity.shape == (1, 1)


def test_pso_build():
    p = pso.PSO()

    assert p.built == True


def test_pso_update_velocity():
    p = pso.PSO()

    velocity = p._update_velocity(1, 1, 1, 1)

    assert velocity != 0


def test_pso_update_position():
    p = pso.PSO()

    position = p._update_position(1, 1)

    assert position == 2
