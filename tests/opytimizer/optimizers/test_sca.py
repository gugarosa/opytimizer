import numpy as np

from opytimizer.core import function
from opytimizer.optimizers import sca
from opytimizer.spaces import search
from opytimizer.utils import constants


def test_sca_hyperparams():
    hyperparams = {
        'r_min': 0,
        'r_max': 2,
        'a': 3,
    }

    new_sca = sca.SCA(hyperparams=hyperparams)

    assert new_sca.r_min == 0

    assert new_sca.r_max == 2

    assert new_sca.a == 3


def test_sca_hyperparams_setter():
    new_sca = sca.SCA()

    try:
        new_sca.r_min = 'a'
    except:
        new_sca.r_min = 0.1

    try:
        new_sca.r_min = -1
    except:
        new_sca.r_min = 0.1

    assert new_sca.r_min == 0.1

    try:
        new_sca.r_max = 'b'
    except:
        new_sca.r_max = 2

    try:
        new_sca.r_max = -1
    except:
        new_sca.r_max = 2

    try:
        new_sca.r_max = 0
    except:
        new_sca.r_max = 2

    assert new_sca.r_max == 2

    try:
        new_sca.a = 'c'
    except:
        new_sca.a = 0.5

    try:
        new_sca.a = -1
    except:
        new_sca.a = 0.5

    assert new_sca.a == 0.5


def test_sca_build():
    new_sca = sca.SCA()

    assert new_sca.built == True


def test_sca_update_position():
    new_sca = sca.SCA()

    position = new_sca._update_position(1, 1, 0.5, 0.5, 0.5, 0.5)

    assert position > 0


def test_sca_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    hyperparams = {
        'r_min': 0,
        'r_max': 2,
        'a': 3
    }

    new_sca = sca.SCA(hyperparams=hyperparams)

    search_space = search.SearchSpace(n_agents=10, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_sca.run(search_space, new_function, pre_evaluation_hook=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, 'The algorithm sca failed to converge.'
