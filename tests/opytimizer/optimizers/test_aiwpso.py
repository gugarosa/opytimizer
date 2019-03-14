import sys

import numpy as np
import pytest
from opytimizer.core import agent, function
from opytimizer.optimizers import aiwpso
from opytimizer.spaces import search


def test_aiwpso_hyperparams():
    hyperparams = {
        'w': 2,
        'w_min': 1,
        'w_max': 3,
        'c1': 1.7,
        'c2': 1.7
    }

    new_aiwpso = aiwpso.AIWPSO(hyperparams=hyperparams)

    assert new_aiwpso.w == 2

    assert new_aiwpso.w_min == 1

    assert new_aiwpso.w_max == 3

    assert new_aiwpso.c1 == 1.7

    assert new_aiwpso.c2 == 1.7


def test_aiwpso_hyperparams_setter():
    new_aiwpso = aiwpso.AIWPSO()

    new_aiwpso.w = 1
    assert new_aiwpso.w == 1

    new_aiwpso.w_min = 0.5
    assert new_aiwpso.w_min == 0.5

    new_aiwpso.w_max = 2
    assert new_aiwpso.w_max == 2

    new_aiwpso.c1 = 1.5
    assert new_aiwpso.c1 == 1.5

    new_aiwpso.c2 = 1.5
    assert new_aiwpso.c2 == 1.5


def test_aiwpso_fitness():
    new_aiwpso = aiwpso.AIWPSO()

    assert new_aiwpso.fitness == None


def test_aiwpso_fitness_setter():
    new_aiwpso = aiwpso.AIWPSO()

    new_aiwpso.fitness = np.zeros(5)

    assert new_aiwpso.fitness.shape == (5, )


def test_aiwpso_local_position():
    new_aiwpso = aiwpso.AIWPSO()

    assert new_aiwpso.local_position == None


def test_aiwpso_local_position_setter():
    new_aiwpso = aiwpso.AIWPSO()

    new_aiwpso.local_position = np.zeros((1, 1))

    assert new_aiwpso.local_position.shape == (1, 1)


def test_aiwpso_velocity():
    new_aiwpso = aiwpso.AIWPSO()

    assert new_aiwpso.velocity == None


def test_aiwpso_velocity_setter():
    new_aiwpso = aiwpso.AIWPSO()

    new_aiwpso.velocity = np.zeros((1, 1))

    assert new_aiwpso.velocity.shape == (1, 1)


def test_aiwpso_build():
    new_aiwpso = aiwpso.AIWPSO()

    assert new_aiwpso.built == True


def test_aiwpso_update_velocity():
    new_aiwpso = aiwpso.AIWPSO()

    velocity = new_aiwpso._update_velocity(1, 1, 1, 1)

    assert velocity != 0


def test_aiwpso_update_position():
    new_aiwpso = aiwpso.AIWPSO()

    position = new_aiwpso._update_position(1, 1)

    assert position == 2

def test_aiwpso_compute_success():
    n_agents = 2

    search_space = search.SearchSpace(n_agents=n_agents, n_iterations=10,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    new_aiwpso = aiwpso.AIWPSO()

    new_fitness = np.zeros(n_agents)

    w = new_aiwpso._compute_success(search_space.agents, new_fitness)

    assert w != 0


def test_aiwpso_evaluate():
    def square(x):
        return np.sum(x**2)

    new_function = function.Function(pointer=square)

    search_space = search.SearchSpace(n_agents=2, n_iterations=10,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    new_aiwpso = aiwpso.AIWPSO()

    local_position = np.zeros((2, 2, 1))

    new_aiwpso._evaluate(search_space, new_function, local_position)

    assert search_space.best_agent.fit < sys.float_info.max


def test_aiwpso_run():
    def square(x):
        return np.sum(x**2)

    new_function = function.Function(pointer=square)

    new_aiwpso = aiwpso.AIWPSO()

    search_space = search.SearchSpace(n_agents=2, n_iterations=10,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_aiwpso.run(search_space, new_function)

    assert len(history.history) > 0
