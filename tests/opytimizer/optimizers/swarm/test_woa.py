import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.swarm import woa
from opytimizer.spaces import search
from opytimizer.utils import constants

np.random.seed(0)


def test_woa_params():
    params = {
        'b': 1
    }

    new_woa = woa.WOA(params=params)

    assert new_woa.b == 1


def test_woa_params_setter():
    new_woa = woa.WOA()

    try:
        new_woa.b = 'a'
    except:
        new_woa.b = 1


def test_woa_build():
    new_woa = woa.WOA()

    assert new_woa.built == True


def test_woa_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    new_woa = woa.WOA()

    search_space = search.SearchSpace(n_agents=10, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_woa.run(search_space, new_function, pre_evaluate=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, 'The algorithm woa failed to converge.'
