import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.swarm import csa
from opytimizer.spaces import search
from opytimizer.utils import constants

np.random.seed(0)


def test_csa_params():
    params = {
        'fl': 2.0,
        'AP': 0.1
    }

    new_csa = csa.CSA(params=params)

    assert new_csa.fl == 2.0

    assert new_csa.AP == 0.1


def test_csa_params_setter():
    new_csa = csa.CSA()

    try:
        new_csa.fl = 'a'
    except:
        new_csa.fl = 2.0

    try:
        new_csa.AP = 'b'
    except:
        new_csa.AP = 0.1

    try:
        new_csa.AP = -1
    except:
        new_csa.AP = 0.1

    assert new_csa.AP == 0.1


def test_csa_build():
    new_csa = csa.CSA()

    assert new_csa.built == True


def test_csa_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    new_csa = csa.CSA()

    search_space = search.SearchSpace(n_agents=10, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_csa.run(search_space, new_function, pre_evaluate=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, 'The algorithm csa failed to converge.'
