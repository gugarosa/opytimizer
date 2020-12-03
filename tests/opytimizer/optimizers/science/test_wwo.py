import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.science import wwo
from opytimizer.spaces import search
from opytimizer.utils import constants

np.random.seed(0)


def test_wwo_hyperparams():
    hyperparams = {
        'h_max': 5,
        'alpha': 1.001,
        'beta': 0.001,
        'k_max': 1
    }

    new_wwo = wwo.WWO(hyperparams=hyperparams)

    assert new_wwo.h_max == 5

    assert new_wwo.alpha == 1.001

    assert new_wwo.beta == 0.001

    assert new_wwo.k_max == 1


def test_wwo_hyperparams_setter():
    new_wwo = wwo.WWO()

    try:
        new_wwo.h_max = 'a'
    except:
        new_wwo.h_max = 5

    try:
        new_wwo.h_max = -1
    except:
        new_wwo.h_max = 5

    assert new_wwo.h_max == 5

    try:
        new_wwo.alpha = 'b'
    except:
        new_wwo.alpha = 1.001

    try:
        new_wwo.alpha = -1
    except:
        new_wwo.alpha = 1.001

    assert new_wwo.alpha == 1.001

    try:
        new_wwo.beta = 'c'
    except:
        new_wwo.beta = 0.001

    try:
        new_wwo.beta = -1
    except:
        new_wwo.beta = 0.001

    assert new_wwo.beta == 0.001

    try:
        new_wwo.k_max = 'd'
    except:
        new_wwo.k_max = 1

    try:
        new_wwo.k_max = -1
    except:
        new_wwo.k_max = 1

    assert new_wwo.k_max == 1


def test_wwo_build():
    new_wwo = wwo.WWO()

    assert new_wwo.built == True


def test_wwo_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    new_wwo = wwo.WWO({'k_max': 20})

    search_space = search.SearchSpace(n_agents=10, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_wwo.run(search_space, new_function, pre_evaluation=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, 'The algorithm wwo failed to converge.'
