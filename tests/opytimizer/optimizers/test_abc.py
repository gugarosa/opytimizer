import numpy as np

from opytimizer.core import function
from opytimizer.optimizers import abc
from opytimizer.spaces import search
from opytimizer.utils import constants


def test_abc_hyperparams():
    hyperparams = {
        'n_trials': 5
    }

    new_abc = abc.ABC(hyperparams=hyperparams)

    assert new_abc.n_trials == 5


def test_abc_hyperparams_setter():
    new_abc = abc.ABC()

    try:
        new_abc.n_trials = 0.0
    except:
        new_abc.n_trials = 10

    try:
        new_abc.n_trials = 0
    except:
        new_abc.n_trials = 10

    assert new_abc.n_trials == 10


def test_abc_build():
    new_abc = abc.ABC()

    assert new_abc.built == True


def test_abc_run():
    def square(x):
        return np.sum(x**2)

    new_function = function.Function(pointer=square)

    hyperparams = {
        'n_trials': 1
    }

    new_abc = abc.ABC(hyperparams=hyperparams)

    search_space = search.SearchSpace(n_agents=10, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_abc.run(search_space, new_function)

    assert len(history.agents) > 0
    assert len(history.best) > 0
    assert len(history.best_index) > 0

    best_fitness = history.best[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, 'The algorithm abc failed to converge.'
