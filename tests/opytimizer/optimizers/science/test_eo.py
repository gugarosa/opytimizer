import numpy as np

from opytimizer.optimizers.science import eo
from opytimizer.spaces import search


def test_eo_params():
    params = {"a1": 2.0, "a2": 1.0, "GP": 0.5, "V": 1.0}

    new_eo = eo.EO(params=params)

    assert new_eo.a1 == 2.0

    assert new_eo.a2 == 1.0

    assert new_eo.GP == 0.5

    assert new_eo.V == 1.0


def test_eo_compile():
    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[1, 1], upper_bound=[10, 10]
    )

    new_eo = eo.EO()
    new_eo.compile(search_space)


def test_eo_calculate_equilibrium():
    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[1, 1], upper_bound=[10, 10]
    )

    new_eo = eo.EO()
    new_eo.compile(search_space)

    new_eo._calculate_equilibrium(search_space.agents)


def test_eo_average_concentration():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[1, 1], upper_bound=[10, 10]
    )

    new_eo = eo.EO()
    new_eo.compile(search_space)

    C_avg = new_eo._average_concentration(square)

    assert type(C_avg).__name__ == "Agent"


def test_eo_update():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[1, 1], upper_bound=[10, 10]
    )

    new_eo = eo.EO()
    new_eo.compile(search_space)

    new_eo.update(search_space, square, 1, 10)
