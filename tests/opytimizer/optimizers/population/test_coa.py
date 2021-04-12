import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.population import coa
from opytimizer.spaces import search
from opytimizer.utils import constants

np.random.seed(0)


def test_coa_hyperparams():
    hyperparams = {
        'n_p': 2
    }

    new_coa = coa.COA(hyperparams=hyperparams)

    assert new_coa.n_p == 2


def test_coa_hyperparams_setter():
    new_coa = coa.COA()

    try:
        new_coa.n_p = 'a'
    except:
        new_coa.n_p = 2


def test_coa_build():
    new_coa = coa.COA()

    assert new_coa.built == True


def test_coa_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    new_coa = coa.COA()

    search_space = search.SearchSpace(n_agents=10, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_coa.run(search_space, new_function, pre_evaluate=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, 'The algorithm coa failed to converge.'
