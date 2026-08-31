import numpy as np

from opytimizer.optimizers.social import mvpa
from opytimizer.spaces import search


def test_mvpa_params():
    params = {"n_teams": 4}

    new_mvpa = mvpa.MVPA(params=params)

    assert new_mvpa.n_teams == 4


def test_mvpa_compile():
    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_mvpa = mvpa.MVPA()
    new_mvpa.compile(search_space)


def test_mvpa_update():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_mvpa = mvpa.MVPA()
    new_mvpa.compile(search_space)

    new_mvpa.update(search_space, square)
    new_mvpa.update(search_space, square)
