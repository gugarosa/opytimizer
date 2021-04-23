import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.population import hho
from opytimizer.spaces import search
from opytimizer.utils import constant

np.random.seed(0)


def test_hho_build():
    new_hho = hho.HHO()

    assert new_hho.built == True


def test_hho_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    new_hho = hho.HHO()

    search_space = search.SearchSpace(n_agents=10, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_hho.run(search_space, new_function, pre_evaluate=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constant.TEST_EPSILON, 'The algorithm hho failed to converge.'
