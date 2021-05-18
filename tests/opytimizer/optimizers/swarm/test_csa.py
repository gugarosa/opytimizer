import numpy as np

from opytimizer.optimizers.swarm import csa
from opytimizer.spaces import search


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


def test_csa_compile():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_csa = csa.CSA()
    new_csa.compile(search_space)

    try:
        new_csa.memory = 1
    except:
        new_csa.memory = np.array([1])

    assert new_csa.memory == 1


def test_csa_evaluate():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_csa = csa.CSA()
    new_csa.compile(search_space)

    new_csa.evaluate(search_space, square)


def test_csa_update():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_csa = csa.CSA()
    new_csa.compile(search_space)

    new_csa.update(search_space)

    new_csa.AP = 1
    new_csa.update(search_space)
