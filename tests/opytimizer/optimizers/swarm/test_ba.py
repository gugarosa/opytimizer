import numpy as np

from opytimizer.optimizers.swarm import ba
from opytimizer.spaces import search


def test_ba_params():
    params = {"f_min": 0, "f_max": 2, "A": 0.5, "r": 0.5}

    new_ba = ba.BA(params=params)

    assert new_ba.f_min == 0

    assert new_ba.f_max == 2

    assert new_ba.A == 0.5

    assert new_ba.r == 0.5


def test_ba_compile():
    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_ba = ba.BA()
    new_ba.compile(search_space)


def test_ba_update():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_ba = ba.BA()
    new_ba.compile(search_space)

    new_ba.update(search_space, square, 1)
