import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.population import aeo
from opytimizer.spaces import search
from opytimizer.utils import constants


def test_aeo_build():
    new_aeo = aeo.AEO()

    assert new_aeo.built == True


def test_aeo_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    new_aeo = aeo.AEO()

    search_space = search.SearchSpace(n_agents=10, n_iterations=30,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_aeo.run(search_space, new_function, pre_evaluation=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, 'The algorithm aeo failed to converge.'
