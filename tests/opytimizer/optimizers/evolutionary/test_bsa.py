import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.evolutionary import bsa
from opytimizer.spaces import search
from opytimizer.utils import constants

np.random.seed(0)


def test_bsa_params():
    params = {
        'F': 3.0,
        'mix_rate': 1
    }

    new_bsa = bsa.BSA(params=params)

    assert new_bsa.F == 3.0

    assert new_bsa.mix_rate == 1


def test_bsa_params_setter():
    new_bsa = bsa.BSA()

    try:
        new_bsa.F = 'a'
    except:
        new_bsa.F = 3.0

    try:
        new_bsa.mix_rate = 'b'
    except:
        new_bsa.mix_rate = 1

    try:
        new_bsa.mix_rate = -1
    except:
        new_bsa.mix_rate = 1

    assert new_bsa.mix_rate == 1


def test_bsa_build():
    new_bsa = bsa.BSA()

    assert new_bsa.built == True


def test_bsa_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    new_bsa = bsa.BSA()

    search_space = search.SearchSpace(n_agents=10, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_bsa.run(search_space, new_function, pre_evaluate=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, 'The algorithm bsa failed to converge.'
