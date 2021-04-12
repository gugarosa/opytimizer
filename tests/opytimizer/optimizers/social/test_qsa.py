import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.social import qsa
from opytimizer.spaces import search
from opytimizer.utils import constants

np.random.seed(0)


def test_qsa_build():
    new_qsa = qsa.QSA()

    assert new_qsa.built == True


def test_qsa_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    new_qsa = qsa.QSA()

    search_space = search.SearchSpace(n_agents=10, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_qsa.run(search_space, new_function, pre_evaluate=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, 'The algorithm qsa failed to converge.'
