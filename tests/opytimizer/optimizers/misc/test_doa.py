import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.misc import doa
from opytimizer.spaces import search
from opytimizer.utils import constants

np.random.seed(0)


def test_doa_hyperparams():
    hyperparams = {
        'r': 1.0
    }

    new_doa = doa.DOA(hyperparams=hyperparams)

    assert new_doa.r == 1.0
    

def test_doa_hyperparams_setter():
    new_doa = doa.DOA()

    try:
        new_doa.r = 'a'
    except:
        new_doa.r = 1.0

    try:
        new_doa.r = -1
    except:
        new_doa.r = 1.0

    assert new_doa.r == 1.0


def test_doa_build():
    new_doa = doa.DOA()

    assert new_doa.built == True


def test_doa_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    new_doa = doa.DOA()

    search_space = search.SearchSpace(n_agents=10, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_doa.run(search_space, new_function, pre_evaluate=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, 'The algorithm doa failed to converge.'
