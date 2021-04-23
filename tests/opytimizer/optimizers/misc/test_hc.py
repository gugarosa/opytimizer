import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.misc import hc
from opytimizer.spaces import search
from opytimizer.utils import constants

np.random.seed(0)


def test_hc_params():
    params = {
        'r_mean': 0,
        'r_var': 0.1,
    }

    new_hc = hc.HC(params=params)

    assert new_hc.r_mean == 0

    assert new_hc.r_var == 0.1


def test_hc_params_setter():
    new_hc = hc.HC()

    try:
        new_hc.r_mean = 'a'
    except:
        new_hc.r_mean = 0.1

    assert new_hc.r_mean == 0.1

    try:
        new_hc.r_var = 'b'
    except:
        new_hc.r_var = 2

    try:
        new_hc.r_var = -1
    except:
        new_hc.r_var = 2

    assert new_hc.r_var == 2


def test_hc_build():
    new_hc = hc.HC()

    assert new_hc.built == True


def test_hc_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    params = {
        'r_mean': 0,
        'r_var': 0.1
    }

    new_hc = hc.HC(params=params)

    search_space = search.SearchSpace(n_agents=50, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_hc.run(search_space, new_function, pre_evaluate=hook)

    history = new_hc.run(search_space, new_function, pre_evaluate=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, 'The algorithm hc failed to converge.'
