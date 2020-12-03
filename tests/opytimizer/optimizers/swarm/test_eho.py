import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.swarm import eho
from opytimizer.spaces import search
from opytimizer.utils import constants

np.random.seed(0)


def test_eho_hyperparams():
    hyperparams = {
        'alpha': 0.5,
        'beta': 0.1,
        'n_clans': 10
    }

    new_eho = eho.EHO(hyperparams=hyperparams)

    assert new_eho.alpha == 0.5

    assert new_eho.beta == 0.1

    assert new_eho.n_clans == 10


def test_eho_hyperparams_setter():
    new_eho = eho.EHO()

    try:
        new_eho.alpha = 'a'
    except:
        new_eho.alpha = 0.5

    try:
        new_eho.alpha = -1
    except:
        new_eho.alpha = 0.5

    assert new_eho.alpha == 0.5

    try:
        new_eho.beta = 'a'
    except:
        new_eho.beta = 0.1

    try:
        new_eho.beta = -1
    except:
        new_eho.beta = 0.1

    assert new_eho.beta == 0.1

    try:
        new_eho.n_clans = 0.0
    except:
        new_eho.n_clans = 10

    try:
        new_eho.n_clans = 0
    except:
        new_eho.n_clans = 10

    assert new_eho.n_clans == 10


def test_eho_build():
    new_eho = eho.EHO()

    assert new_eho.built == True


def test_eho_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    new_eho = eho.EHO()

    

    try:
        search_space = search.SearchSpace(n_agents=5, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

        history = new_eho.run(search_space, new_function, pre_evaluation=hook)

    except:
        search_space = search.SearchSpace(n_agents=20, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

        history = new_eho.run(search_space, new_function, pre_evaluation=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, 'The algorithm eho failed to converge.'
