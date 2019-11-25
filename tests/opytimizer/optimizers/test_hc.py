import numpy as np

from opytimizer.core import function
from opytimizer.optimizers import hc
from opytimizer.spaces import search
from opytimizer.utils import constants


def test_hc_hyperparams():
    hyperparams = {
        'type': 'gaussian',
        'r_min': 0,
        'r_max': 0.1,
    }

    new_hc = hc.HC(hyperparams=hyperparams)

    assert new_hc.type == 'gaussian'

    assert new_hc.r_min == 0

    assert new_hc.r_max == 0.1


def test_hc_hyperparams_setter():
    new_hc = hc.HC()

    try:
        new_hc.type = 'g'
    except:
        new_hc.type = 'gaussian'

    assert new_hc.type == 'gaussian'

    try:
        new_hc.r_min = 'b'
    except:
        new_hc.r_min = 0.1

    assert new_hc.r_min == 0.1

    try:
        new_hc.r_max = 'c'
    except:
        new_hc.r_max = 2

    try:
        new_hc.r_max = -1
    except:
        new_hc.r_max = 2

    try:
        new_hc.r_max = 0
    except:
        new_hc.r_max = 2

    assert new_hc.r_max == 2


def test_hc_build():
    new_hc = hc.HC()

    assert new_hc.built == True


def test_hc_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    hyperparams = {
        'type': 'gaussian',
        'r_min': 0,
        'r_max': 0.1
    }

    new_hc = hc.HC(hyperparams=hyperparams)

    search_space = search.SearchSpace(n_agents=50, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_hc.run(search_space, new_function, pre_evaluation_hook=hook)

    hyperparams = {
        'type': 'uniform',
        'r_min': 0,
        'r_max': 0.1
    }

    new_hc = hc.HC(hyperparams=hyperparams)

    history = new_hc.run(search_space, new_function, pre_evaluation_hook=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, 'The algorithm hc failed to converge.'
