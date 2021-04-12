import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.science import sa
from opytimizer.spaces import search
from opytimizer.utils import constants

np.random.seed(0)


def test_sa_hyperparams():
    hyperparams = {
        'T': 100,
        'beta': 0.99,
    }

    new_sa = sa.SA(hyperparams=hyperparams)

    assert new_sa.T == 100

    assert new_sa.beta == 0.99


def test_sa_hyperparams_setter():
    new_sa = sa.SA()

    try:
        new_sa.T = 'a'
    except:
        new_sa.T = 10

    try:
        new_sa.T = -1
    except:
        new_sa.T = 10

    assert new_sa.T == 10

    try:
        new_sa.beta = 'b'
    except:
        new_sa.beta = 0.5

    try:
        new_sa.beta = -1
    except:
        new_sa.beta = 0.5

    assert new_sa.beta == 0.5


def test_sa_build():
    new_sa = sa.SA()

    assert new_sa.built == True


def test_sa_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    hyperparams = {
        'T': 100,
        'beta': 0.99
    }

    new_sa = sa.SA(hyperparams=hyperparams)

    search_space = search.SearchSpace(n_agents=10, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_sa.run(search_space, new_function, pre_evaluate=hook)

    history = new_sa.run(search_space, new_function, pre_evaluate=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, 'The algorithm sa failed to converge.'
