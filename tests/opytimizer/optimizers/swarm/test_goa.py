import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.swarm import goa
from opytimizer.spaces import search
from opytimizer.utils import constant

np.random.seed(0)


def test_goa_params():
    params = {
        'c_min': 0.00001,
        'c_max': 1.0,
        'f': 0.5,
        'l': 1.5
    }

    new_goa = goa.GOA(params=params)

    assert new_goa.c_min == 0.00001

    assert new_goa.c_max == 1.0

    assert new_goa.f == 0.5

    assert new_goa.l == 1.5


def test_goa_params_setter():
    new_goa = goa.GOA()

    try:
        new_goa.c_min = 'a'
    except:
        new_goa.c_min = 0.00001

    try:
        new_goa.c_min = -1
    except:
        new_goa.c_min = 0.00001

    assert new_goa.c_min == 0.00001

    try:
        new_goa.c_max = 'b'
    except:
        new_goa.c_max = 2.0

    try:
        new_goa.c_max = 0
    except:
        new_goa.c_max = 1.0

    assert new_goa.c_max == 1.0

    try:
        new_goa.f = 'c'
    except:
        new_goa.f = 0.5

    try:
        new_goa.f = -1
    except:
        new_goa.f = 0.5

    assert new_goa.f == 0.5

    try:
        new_goa.l = 'd'
    except:
        new_goa.l = 1.5

    try:
        new_goa.l = -1
    except:
        new_goa.l = 1.5

    assert new_goa.l == 1.5


def test_goa_build():
    new_goa = goa.GOA()

    assert new_goa.built == True


def test_goa_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    new_goa = goa.GOA()

    search_space = search.SearchSpace(n_agents=10, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_goa.run(search_space, new_function, pre_evaluate=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constant.TEST_EPSILON, 'The algorithm goa failed to converge.'
