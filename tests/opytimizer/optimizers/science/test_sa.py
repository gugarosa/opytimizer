import numpy as np

from opytimizer.optimizers.science import sa
from opytimizer.spaces import search


def test_sa_params():
    params = {
        'T': 100,
        'beta': 0.99,
    }

    new_sa = sa.SA(params=params)

    assert new_sa.T == 100

    assert new_sa.beta == 0.99


def test_sa_params_setter():
    new_sa = sa.SA()

    try:
        new_sa.T = 'a'
    except:
        new_sa.T = 10

    try:
        new_sa.T = -1
    except:
        new_sa.T = 10

    assert new_sa.T == 10

    try:
        new_sa.beta = 'b'
    except:
        new_sa.beta = 0.5

    try:
        new_sa.beta = -1
    except:
        new_sa.beta = 0.5

    assert new_sa.beta == 0.5


def test_sa_update():
    def square(x):
        return np.sum(x**2)

    new_sa = sa.SA()

    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_sa.update(search_space, square)
    new_sa.update(search_space, square)
