import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.swarm import bwo
from opytimizer.spaces import search
from opytimizer.utils import constant

np.random.seed(0)


def test_bwo_params():
    params = {
        'pp': 0.6,
        'cr': 0.44,
        'pm': 0.4,
    }

    new_bwo = bwo.BWO(params=params)

    assert new_bwo.pp == 0.6

    assert new_bwo.cr == 0.44

    assert new_bwo.pm == 0.4


def test_bwo_params_setter():
    new_bwo = bwo.BWO()

    try:
        new_bwo.pp = 'a'
    except:
        new_bwo.pp = 0.6

    try:
        new_bwo.pp = -1
    except:
        new_bwo.pp = 0.6

    assert new_bwo.pp == 0.6

    try:
        new_bwo.cr = 'b'
    except:
        new_bwo.cr = 0.44

    try:
        new_bwo.cr = -1
    except:
        new_bwo.cr = 0.44

    assert new_bwo.cr == 0.44

    try:
        new_bwo.pm = 'c'
    except:
        new_bwo.pm = 0.4

    try:
        new_bwo.pm = -1
    except:
        new_bwo.pm = 0.4

    assert new_bwo.pm == 0.4


def test_bwo_build():
    new_bwo = bwo.BWO()

    assert new_bwo.built == True


def test_bwo_update():
    def square(x):
        return np.sum(x**2)

    new_function = function.Function(pointer=square)

    new_bwo = bwo.BWO()

    search_space = search.SearchSpace(n_agents=10, n_iterations=10,
                                      n_variables=2, lower_bound=[1, 1],
                                      upper_bound=[10, 10])

    new_bwo._evaluate(search_space, new_function)

    new_bwo._update(search_space.agents,
                    search_space.n_variables, new_function)

    assert search_space.agents[0].position[0] != 0


def test_bwo_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    new_bwo = bwo.BWO()

    search_space = search.SearchSpace(n_agents=10, n_iterations=30,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_bwo.run(search_space, new_function, pre_evaluate=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constant.TEST_EPSILON, 'The algorithm bwo failed to converge.'
