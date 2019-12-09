import sys

import numpy as np

from opytimizer.core import function
from opytimizer.optimizers import rpso
from opytimizer.spaces import search
from opytimizer.utils import constants


def test_rpso_hyperparams():
    hyperparams = {
        'c1': 1.7,
        'c2': 1.7
    }

    new_rpso = rpso.RPSO(hyperparams=hyperparams)

    assert new_rpso.c1 == 1.7
    assert new_rpso.c2 == 1.7


def test_rpso_hyperparams_setter():
    new_rpso = rpso.RPSO()

    try:
        new_rpso.c1 = 'a'
    except:
        new_rpso.c1 = 1.5

    try:
        new_rpso.c1 = -1
    except:
        new_rpso.c1 = 1.5

    assert new_rpso.c1 == 1.5

    try:
        new_rpso.c2 = 'b'
    except:
        new_rpso.c2 = 1.5

    try:
        new_rpso.c2 = -1
    except:
        new_rpso.c2 = 1.5

    assert new_rpso.c2 == 1.5


def test_rpso_build():
    new_rpso = rpso.RPSO()

    assert new_rpso.built == True


def test_rpso_update_velocity():
    new_rpso = rpso.RPSO()

    velocity = new_rpso._update_velocity(1, 1, 1, 10, 1, 1)

    assert velocity != 0


def test_rpso_update_position():
    new_rpso = rpso.RPSO()

    position = new_rpso._update_position(1, 1)

    assert position == 2


def test_rpso_evaluate():
    def square(x):
        return np.sum(x**2)

    new_function = function.Function(pointer=square)

    search_space = search.SearchSpace(n_agents=2, n_iterations=10,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    new_rpso = rpso.RPSO()

    local_position = np.zeros((2, 2, 1))

    new_rpso._evaluate(search_space, new_function, local_position)

    assert search_space.best_agent.fit < sys.float_info.max


def test_rpso_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    new_rpso = rpso.RPSO()

    search_space = search.SearchSpace(n_agents=5, n_iterations=20,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_rpso.run(search_space, new_function, pre_evaluation_hook=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0
    assert len(history.local) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, 'The algorithm rpso failed to converge.'
