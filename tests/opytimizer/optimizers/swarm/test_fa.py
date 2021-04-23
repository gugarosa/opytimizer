import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.swarm import fa
from opytimizer.spaces import search
from opytimizer.utils import constants

np.random.seed(0)


def test_fa_params():
    params = {
        'alpha': 0.5,
        'beta': 0.2,
        'gamma': 1.0
    }

    new_fa = fa.FA(params=params)

    assert new_fa.alpha == 0.5

    assert new_fa.beta == 0.2

    assert new_fa.gamma == 1.0


def test_fa_params_setter():
    new_fa = fa.FA()

    try:
        new_fa.alpha = 'a'
    except:
        new_fa.alpha = 0.5

    try:
        new_fa.alpha = -1
    except:
        new_fa.alpha = 0.5

    assert new_fa.alpha == 0.5

    try:
        new_fa.beta = 'b'
    except:
        new_fa.beta = 0.2

    try:
        new_fa.beta = -1
    except:
        new_fa.beta = 0.2

    assert new_fa.beta == 0.2

    try:
        new_fa.gamma = 'c'
    except:
        new_fa.gamma = 1.0

    try:
        new_fa.gamma = -1
    except:
        new_fa.gamma = 1.0

    assert new_fa.gamma == 1.0


def test_fa_build():
    new_fa = fa.FA()

    assert new_fa.built == True


def test_fa_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    new_fa = fa.FA()

    search_space = search.SearchSpace(n_agents=10, n_iterations=10,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_fa.run(search_space, new_function, pre_evaluate=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, 'The algorithm fa failed to converge.'
