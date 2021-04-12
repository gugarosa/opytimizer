import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.misc import cem
from opytimizer.spaces import search
from opytimizer.utils import constants

np.random.seed(0)


def test_cem_hyperparams():
    hyperparams = {
        'n_updates': 5,
        'alpha': 0.7,
    }

    new_cem = cem.CEM(hyperparams=hyperparams)

    assert new_cem.n_updates == 5

    assert new_cem.alpha == 0.7


def test_cem_hyperparams_setter():
    new_cem = cem.CEM()

    try:
        new_cem.n_updates = 'a'
    except:
        new_cem.n_updates = 10

    try:
        new_cem.n_updates = -1
    except:
        new_cem.n_updates = 10

    assert new_cem.n_updates == 10

    try:
        new_cem.alpha = 'b'
    except:
        new_cem.alpha = 0.5

    try:
        new_cem.alpha = -1
    except:
        new_cem.alpha = 0.5

    assert new_cem.alpha == 0.5


def test_cem_build():
    new_cem = cem.CEM()

    assert new_cem.built == True


def test_cem_create_new_samples():
    def square(x):
        return np.sum(x**2)

    new_cem = cem.CEM()

    search_space = search.SearchSpace(n_agents=10, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    new_cem._create_new_samples(
        search_space.agents, square, np.array([1, 1]), np.array([1, 1]))



def test_cem_update_mean():
    new_cem = cem.CEM()

    mean = new_cem._update_mean(np.array([1, 1]), 1)

    assert mean != 0


def test_cem_update_std():
    new_cem = cem.CEM()

    std = new_cem._update_std(np.array([1, 1]), 1, 0.25)

    assert std != 0


def test_cem_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    hyperparams = {
        'n_updates': 5,
        'alpha': 0.7
    }

    new_cem = cem.CEM(hyperparams=hyperparams)

    search_space = search.SearchSpace(n_agents=10, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_cem.run(search_space, new_function, pre_evaluate=hook)

    history = new_cem.run(search_space, new_function, pre_evaluate=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, 'The algorithm cem failed to converge.'
