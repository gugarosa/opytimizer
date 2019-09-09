import numpy as np

from opytimizer.core import function
from opytimizer.math import constants
from opytimizer.optimizers import fa
from opytimizer.spaces import search


def test_fa_hyperparams():
    hyperparams = {
        'alpha': 0.5,
        'beta': 0.2,
        'gamma': 1.0
    }

    new_fa = fa.FA(hyperparams=hyperparams)

    assert new_fa.alpha == 0.5

    assert new_fa.beta == 0.2

    assert new_fa.gamma == 1.0


def test_fa_hyperparams_setter():
    new_fa = fa.FA()

    new_fa.alpha = 0.25
    assert new_fa.alpha == 0.25

    new_fa.beta = 1.0
    assert new_fa.beta == 1.0

    new_fa.gamma = 1.5
    assert new_fa.gamma == 1.5


def test_fa_build():
    new_fa = fa.FA()

    assert new_fa.built == True


def test_fa_run():
    def square(x):
        return np.sum(x**2)

    new_function = function.Function(pointer=square)

    new_fa = fa.FA()

    search_space = search.SearchSpace(n_agents=10, n_iterations=10,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_fa.run(search_space, new_function)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, "The algorithm fa failed to converge"
