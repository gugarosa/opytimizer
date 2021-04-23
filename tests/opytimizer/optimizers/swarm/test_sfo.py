import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.swarm import sfo
from opytimizer.spaces import search
from opytimizer.utils import constant

np.random.seed(0)


def test_sfo_params():
    params = {
        'PP': 0.1,
        'A': 4,
        'e': 0.001
    }

    new_sfo = sfo.SFO(params=params)

    assert new_sfo.PP == 0.1

    assert new_sfo.A == 4

    assert new_sfo.e == 0.001


def test_sfo_params_setter():
    new_sfo = sfo.SFO()

    try:
        new_sfo.PP = 'a'
    except:
        new_sfo.PP = 0.1

    try:
        new_sfo.PP = -1
    except:
        new_sfo.PP = 0.1

    assert new_sfo.PP == 0.1

    try:
        new_sfo.A = 'b'
    except:
        new_sfo.A = 4

    try:
        new_sfo.A = 0
    except:
        new_sfo.A = 4

    assert new_sfo.A == 4

    try:
        new_sfo.e = 'c'
    except:
        new_sfo.e = 0.001

    try:
        new_sfo.e = -1
    except:
        new_sfo.e = 0.001

    assert new_sfo.e == 0.001


def test_sfo_build():
    new_sfo = sfo.SFO()

    assert new_sfo.built == True


def test_sfo_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    new_sfo = sfo.SFO()

    search_space = search.SearchSpace(n_agents=10, n_iterations=500,
                                      n_variables=3, lower_bound=[0, 0, 0],
                                      upper_bound=[10, 10, 10])

    history = new_sfo.run(search_space, new_function, pre_evaluate=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constant.TEST_EPSILON, 'The algorithm sfo failed to converge.'
