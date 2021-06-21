import numpy as np

from opytimizer.optimizers.swarm import fso
from opytimizer.spaces import search


def test_fso_params():
    params = {
        'beta': 0.5,
    }

    new_fso = fso.FSO(params=params)

    assert new_fso.beta == 0.5


def test_fso_params_setter():
    new_fso = fso.FSO()

    try:
        new_fso.beta = 'a'
    except:
        new_fso.beta = 0.5

    assert new_fso.beta == 0.5

    try:
        new_fso.beta = -1
    except:
        new_fso.beta = 0.5

    assert new_fso.beta == 0.5


def test_fso_update():
    def square(x):
        return np.sum(x**2)

    new_fso = fso.FSO()

    search_space = search.SearchSpace(n_agents=2, n_variables=2,
                                      lower_bound=[1, 1], upper_bound=[10, 10])

    new_fso.update(search_space, square, 1, 10)
