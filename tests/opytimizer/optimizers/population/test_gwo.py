import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.population import gwo
from opytimizer.spaces import search
from opytimizer.utils import constants

np.random.seed(0)


def test_gwo_build():
    new_gwo = gwo.GWO()

    assert new_gwo.built == True


def test_gwo_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    new_gwo = gwo.GWO()

    search_space = search.SearchSpace(n_agents=10, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_gwo.run(search_space, new_function, pre_evaluate=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, 'The algorithm gwo failed to converge.'
