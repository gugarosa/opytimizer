import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.science import aso
from opytimizer.spaces import search
from opytimizer.utils import constants

np.random.seed(0)


def test_aso_hyperparams():
    hyperparams = {
        'alpha': 50.0,
        'beta': 0.2
    }

    new_aso = aso.ASO(hyperparams=hyperparams)

    assert new_aso.alpha == 50.0

    assert new_aso.beta == 0.2


def test_aso_hyperparams_setter():
    new_aso = aso.ASO()

    try:
        new_aso.alpha = 'a'
    except:
        new_aso.alpha = 50.0

    try:
        new_aso.beta = 'b'
    except:
        new_aso.beta = 0.2

    try:
        new_aso.beta = -1
    except:
        new_aso.beta = 0.2

    assert new_aso.beta == 0.2


def test_aso_build():
    new_aso = aso.ASO()

    assert new_aso.built == True


def test_aso_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    new_aso = aso.ASO()

    search_space = search.SearchSpace(n_agents=10, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_aso.run(search_space, new_function, pre_evaluate=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, 'The algorithm aso failed to converge.'
