import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.evolutionary import iwo
from opytimizer.spaces import search
from opytimizer.utils import constant

np.random.seed(0)


def test_iwo_params():
    params = {
        'min_seeds': 0,
        'max_seeds': 5,
        'e': 2,
        'final_sigma': 0.001,
        'init_sigma': 3
    }

    new_iwo = iwo.IWO(params=params)

    assert new_iwo.min_seeds == 0

    assert new_iwo.max_seeds == 5

    assert new_iwo.e == 2

    assert new_iwo.final_sigma == 0.001

    assert new_iwo.init_sigma == 3


def test_iwo_params_setter():
    new_iwo = iwo.IWO()

    try:
        new_iwo.min_seeds = 'a'
    except:
        new_iwo.min_seeds = 0

    try:
        new_iwo.min_seeds = -1
    except:
        new_iwo.min_seeds = 0

    assert new_iwo.min_seeds == 0

    try:
        new_iwo.max_seeds = 'b'
    except:
        new_iwo.max_seeds = 2

    try:
        new_iwo.max_seeds = -1
    except:
        new_iwo.max_seeds = 2

    assert new_iwo.max_seeds == 2

    try:
        new_iwo.e = 'c'
    except:
        new_iwo.e = 1.5

    try:
        new_iwo.e = -1
    except:
        new_iwo.e = 1.5

    assert new_iwo.e == 1.5

    try:
        new_iwo.final_sigma = 'd'
    except:
        new_iwo.final_sigma = 1.5

    try:
        new_iwo.final_sigma = -1
    except:
        new_iwo.final_sigma = 1.5

    assert new_iwo.final_sigma == 1.5

    try:
        new_iwo.init_sigma = 'e'
    except:
        new_iwo.init_sigma = 2.0

    try:
        new_iwo.init_sigma = -1
    except:
        new_iwo.init_sigma = 2.0

    try:
        new_iwo.init_sigma = 1.3
    except:
        new_iwo.init_sigma = 2.0

    assert new_iwo.init_sigma == 2.0


def test_iwo_build():
    new_iwo = iwo.IWO()

    assert new_iwo.built == True


def test_iwo_update():
    def square(x):
        return np.sum(x**2)

    assert square(2) == 4

    new_function = function.Function(pointer=square)

    new_iwo = iwo.IWO()

    search_space = search.SearchSpace(n_agents=2, n_iterations=10,
                                      n_variables=2, lower_bound=[1, 1],
                                      upper_bound=[10, 10])

    new_iwo._update(search_space.agents, search_space.n_agents, new_function)

    assert search_space.agents[0].position[0] != 0


def test_iwo_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    new_iwo = iwo.IWO()

    search_space = search.SearchSpace(n_agents=10, n_iterations=30,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_iwo.run(search_space, new_function, pre_evaluate=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constant.TEST_EPSILON, 'The algorithm iwo failed to converge.'
