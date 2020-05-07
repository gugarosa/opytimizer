import sys

import numpy as np

import opytimizer.math.random as r
from opytimizer.core import function
from opytimizer.optimizers.boolean import bpso
from opytimizer.spaces import search
from opytimizer.utils import constants


def test_bpso_hyperparams():
    hyperparams = {
        'c1': r.generate_binary_random_number(size=(1, 1)),
        'c2': r.generate_binary_random_number(size=(1, 1))
    }

    new_bpso = bpso.BPSO(hyperparams=hyperparams)

    assert new_bpso.c1 == 0 or new_bpso.c1 == 1
    
    assert new_bpso.c2 == 0 or new_bpso.c2 == 1


def test_bpso_hyperparams_setter():
    new_bpso = bpso.BPSO()

    try:
        new_bpso.c1 = 'a'
    except:
        new_bpso.c1 = r.generate_binary_random_number(size=(1, 1))

    assert new_bpso.c1 == 0 or new_bpso.c1 == 1

    try:
        new_bpso.c2 = 'b'
    except:
        new_bpso.c2 = r.generate_binary_random_number(size=(1, 1))

    assert new_bpso.c2 == 0 or new_bpso.c2 == 1


def test_bpso_build():
    new_bpso = bpso.BPSO()

    assert new_bpso.built == True


def test_bpso_update_velocity():
    new_bpso = bpso.BPSO()

    velocity = new_bpso._update_velocity(
        np.array([1]), np.array([1]), np.array([1]))

    assert velocity == 0 or velocity == 1


def test_bpso_update_position():
    new_bpso = bpso.BPSO()

    position = new_bpso._update_position(1, 1)

    assert position == 0 or position == 1


def test_bpso_evaluate():
    def square(x):
        return np.sum(x**2)

    new_function = function.Function(pointer=square)

    search_space = search.SearchSpace(n_agents=2, n_iterations=10,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    new_bpso = bpso.BPSO()

    local_position = np.zeros((2, 2, 1))

    new_bpso._evaluate(search_space, new_function, local_position)

    assert search_space.best_agent.fit < sys.float_info.max


def test_bpso_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    new_bpso = bpso.BPSO()

    search_space = search.SearchSpace(n_agents=5, n_iterations=20,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_bpso.run(search_space, new_function, pre_evaluation=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0
    assert len(history.local) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, 'The algorithm bpso failed to converge.'
