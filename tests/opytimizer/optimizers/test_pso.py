import sys

import numpy as np
import pytest
from opytimizer.core import function
from opytimizer.optimizers import pso
from opytimizer.spaces import search


def test_pso_hyperparams():
    hyperparams = {
        'w': 2,
        'c1': 1.7,
        'c2': 1.7
    }

    new_pso = pso.PSO(hyperparams=hyperparams)

    assert new_pso.w == 2

    assert new_pso.c1 == 1.7

    assert new_pso.c2 == 1.7


def test_pso_hyperparams_setter():
    new_pso = pso.PSO()

    new_pso.w = 1
    assert new_pso.w == 1

    new_pso.c1 = 1.5
    assert new_pso.c1 == 1.5

    new_pso.c2 = 1.5
    assert new_pso.c2 == 1.5


def test_pso_local_position():
    new_pso = pso.PSO()

    assert new_pso.local_position == None


def test_pso_local_position_setter():
    new_pso = pso.PSO()

    new_pso.local_position = np.zeros((1, 1))

    assert new_pso.local_position.shape == (1, 1)


def test_pso_velocity():
    new_pso = pso.PSO()

    assert new_pso.velocity == None


def test_pso_velocity_setter():
    new_pso = pso.PSO()

    new_pso.velocity = np.zeros((1, 1))

    assert new_pso.velocity.shape == (1, 1)


def test_pso_build():
    new_pso = pso.PSO()

    assert new_pso.built == True


def test_pso_update_velocity():
    new_pso = pso.PSO()

    velocity = new_pso._update_velocity(1, 1, 1, 1)

    assert velocity != 0


def test_pso_update_position():
    new_pso = pso.PSO()

    position = new_pso._update_position(1, 1)

    assert position == 2


def test_pso_evaluate():
    def square(x):
        return np.sum(x**2)

    new_function = function.Function(pointer=square)

    search_space = search.SearchSpace(n_agents=2, n_iterations=10,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    new_pso = pso.PSO()

    local_position = np.zeros((2, 2, 1))

    new_pso._evaluate(search_space, new_function, local_position)

    assert search_space.best_agent.fit < sys.float_info.max


def test_pso_run():
    def square(x):
        return np.sum(x**2)

    new_function = function.Function(pointer=square)

    new_pso = pso.PSO()

    search_space = search.SearchSpace(n_agents=2, n_iterations=10,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_pso.run(search_space, new_function)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0
