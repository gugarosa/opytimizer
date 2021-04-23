import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.population import ao
from opytimizer.spaces import search
from opytimizer.utils import constant

np.random.seed(0)


def test_ao_params():
    params = {
        'alpha': 0.1,
        'delta': 0.1,
        'n_cycles': 10,
        'U': 0.00565,
        'w': 0.005
    }

    new_ao = ao.AO(params=params)

    assert new_ao.alpha == 0.1

    assert new_ao.delta == 0.1

    assert new_ao.n_cycles == 10

    assert new_ao.U == 0.00565

    assert new_ao.w == 0.005


def test_ao_params_setter():
    new_ao = ao.AO()

    try:
        new_ao.alpha = 'a'
    except:
        new_ao.alpha = 0.1

    try:
        new_ao.alpha = -1
    except:
        new_ao.alpha = 0.1

    assert new_ao.alpha == 0.1

    try:
        new_ao.delta = 'b'
    except:
        new_ao.delta = 0.1

    try:
        new_ao.delta = -1
    except:
        new_ao.delta = 0.1

    try:
        new_ao.n_cycles = 'c'
    except:
        new_ao.n_cycles = 10

    try:
        new_ao.n_cycles = -1
    except:
        new_ao.n_cycles = 10

    assert new_ao.n_cycles == 10

    try:
        new_ao.U = 'd'
    except:
        new_ao.U = 0.00565

    try:
        new_ao.U = -1
    except:
        new_ao.U = 0.00565

    assert new_ao.U == 0.00565

    try:
        new_ao.U = 'e'
    except:
        new_ao.U = 0.005

    try:
        new_ao.U = -1
    except:
        new_ao.U = 0.005

    assert new_ao.U == 0.005


def test_ao_build():
    new_ao = ao.AO()

    assert new_ao.built == True


def test_ao_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    params = {
        'alpha': 0.1,
        'delta': 0.1,
        'n_cycles': 5,
        'U': 0.00565,
        'w': 0.005
    }

    new_ao = ao.AO(params=params)

    search_space = search.SearchSpace(n_agents=10, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_ao.run(search_space, new_function, pre_evaluate=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constant.TEST_EPSILON, 'The algorithm ao failed to converge.'
