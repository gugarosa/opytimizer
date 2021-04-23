import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.misc import aoa
from opytimizer.spaces import search
from opytimizer.utils import constant

np.random.seed(0)


def test_aoa_params():
    params = {
        'a_min': 0.2,
        'a_max': 1.0,
        'alpha': 5.0,
        'mu': 0.499
    }

    new_aoa = aoa.AOA(params=params)

    assert new_aoa.a_min == 0.2

    assert new_aoa.a_max == 1.0

    assert new_aoa.alpha == 5.0

    assert new_aoa.mu == 0.499


def test_aoa_params_setter():
    new_aoa = aoa.AOA()

    try:
        new_aoa.a_min = 'a'
    except:
        new_aoa.a_min = 0.2

    try:
        new_aoa.a_min = -1
    except:
        new_aoa.a_min = 0.2

    assert new_aoa.a_min == 0.2

    try:
        new_aoa.a_max = 'b'
    except:
        new_aoa.a_max = 1.0

    try:
        new_aoa.a_max = -1
    except:
        new_aoa.a_max = 1.0

    try:
        new_aoa.a_max = 0
    except:
        new_aoa.a_max = 1.0

    assert new_aoa.a_max == 1.0

    try:
        new_aoa.alpha = 'c'
    except:
        new_aoa.alpha = 5.0

    try:
        new_aoa.alpha = -1
    except:
        new_aoa.alpha = 5.0

    assert new_aoa.alpha == 5.0

    try:
        new_aoa.mu = 'd'
    except:
        new_aoa.mu = 0.499

    try:
        new_aoa.mu = -1
    except:
        new_aoa.mu = 0.499

    assert new_aoa.mu == 0.499


def test_aoa_build():
    new_aoa = aoa.AOA()

    assert new_aoa.built == True


def test_aoa_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    params = {
        'a_min': 0.2,
        'a_max': 1.0,
        'alpha': 5,
        'mu': 0.499
    }

    new_aoa = aoa.AOA(params=params)

    search_space = search.SearchSpace(n_agents=10, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_aoa.run(search_space, new_function, pre_evaluate=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constant.TEST_EPSILON, 'The algorithm aoa failed to converge.'
