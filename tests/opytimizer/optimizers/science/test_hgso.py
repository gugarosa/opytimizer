import numpy as np

from opytimizer.optimizers.science import hgso
from opytimizer.spaces import search

np.random.seed(0)


def test_hgso_params():
    params = {
        "n_clusters": 2,
        "l1": 0.0005,
        "l2": 100,
        "l3": 0.001,
        "alpha": 1.0,
        "beta": 1.0,
        "K": 1.0,
    }

    new_hgso = hgso.HGSO(params=params)

    assert new_hgso.n_clusters == 2

    assert new_hgso.l1 == 0.0005

    assert new_hgso.l2 == 100

    assert new_hgso.l3 == 0.001

    assert new_hgso.alpha == 1.0

    assert new_hgso.beta == 1.0

    assert new_hgso.K == 1.0


def test_hgso_compile():
    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_hgso = hgso.HGSO()
    new_hgso.compile(search_space)


def test_hgso_update_position():
    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_hgso = hgso.HGSO()
    new_hgso.compile(search_space)

    position = new_hgso._update_position(
        search_space.agents[0], search_space.agents[1], search_space.best_agent, 0.5
    )

    assert position[0][0] != 0


def test_hgso_update():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_hgso = hgso.HGSO()
    new_hgso.compile(search_space)

    new_hgso.update(search_space, square, 1, 10)
