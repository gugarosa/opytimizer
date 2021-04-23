import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.swarm import mfo
from opytimizer.spaces import search
from opytimizer.utils import constant

np.random.seed(0)


def test_mfo_params():
    params = {
        'b': 1
    }

    new_mfo = mfo.MFO(params=params)

    assert new_mfo.b == 1


def test_mfo_params_setter():
    new_mfo = mfo.MFO()

    try:
        new_mfo.b = 'a'
    except:
        new_mfo.b = 1

    try:
        new_mfo.b = -1
    except:
        new_mfo.b = 1

    assert new_mfo.b == 1


def test_mfo_build():
    new_mfo = mfo.MFO()

    assert new_mfo.built == True


def test_mfo_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    new_mfo = mfo.MFO()

    search_space = search.SearchSpace(n_agents=10, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_mfo.run(search_space, new_function, pre_evaluate=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constant.TEST_EPSILON, 'The algorithm mfo failed to converge.'
