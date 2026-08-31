import numpy as np

from opytimizer.optimizers.misc import cem
from opytimizer.spaces import search


def test_cem_params():
    params = {
        "n_updates": 5,
        "alpha": 0.7,
    }

    new_cem = cem.CEM(params=params)

    assert new_cem.n_updates == 5

    assert new_cem.alpha == 0.7


def test_cem_compile():
    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_cem = cem.CEM()
    new_cem.compile(search_space)


def test_cem_create_new_samples():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_cem = cem.CEM()
    new_cem.compile(search_space)

    new_cem._create_new_samples(search_space.agents, square)


def test_cem_update_mean():
    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_cem = cem.CEM()
    new_cem.compile(search_space)

    new_cem._update_mean(np.array([1, 1]))

    assert new_cem.mean[0] != 0


def test_cem_update_std():
    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_cem = cem.CEM()
    new_cem.compile(search_space)

    new_cem._update_std(np.array([1, 1]))

    assert new_cem.std[0] != 0


def test_cem_update():
    def square(x):
        return np.sum(x**2)

    new_function = square

    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_cem = cem.CEM()
    new_cem.compile(search_space)

    new_cem.update(search_space, new_function)
