import numpy as np

from opytimizer.optimizers.swarm import eho
from opytimizer.spaces import search


def test_eho_params():
    params = {"alpha": 0.5, "beta": 0.1, "n_clans": 10}

    new_eho = eho.EHO(params=params)

    assert new_eho.alpha == 0.5

    assert new_eho.beta == 0.1

    assert new_eho.n_clans == 10


def test_eho_params_setter():
    new_eho = eho.EHO()

    try:
        new_eho.alpha = "a"
    except:
        new_eho.alpha = 0.5

    try:
        new_eho.alpha = -1
    except:
        new_eho.alpha = 0.5

    assert new_eho.alpha == 0.5

    try:
        new_eho.beta = "a"
    except:
        new_eho.beta = 0.1

    try:
        new_eho.beta = -1
    except:
        new_eho.beta = 0.1

    assert new_eho.beta == 0.1

    try:
        new_eho.n_clans = 0.0
    except:
        new_eho.n_clans = 10

    try:
        new_eho.n_clans = 0
    except:
        new_eho.n_clans = 10

    assert new_eho.n_clans == 10


def test_eho_compile():
    search_space = search.SearchSpace(
        n_agents=20, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_eho = eho.EHO()
    new_eho.compile(search_space)

    try:
        new_eho.n_ci = "a"
    except:
        new_eho.n_ci = 1

    assert new_eho.n_ci == 1

    try:
        new_eho.n_ci = -1
    except:
        new_eho.n_ci = 1

    assert new_eho.n_ci == 1


def test_eho_get_agents_from_clan():
    search_space = search.SearchSpace(
        n_agents=20, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_eho = eho.EHO()
    new_eho.compile(search_space)

    agents = new_eho._get_agents_from_clan(search_space.agents, 0)

    assert len(agents) == 2

    agents = new_eho._get_agents_from_clan(search_space.agents, 9)

    assert len(agents) == 2


def test_eho_updating_operator():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(
        n_agents=20, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_eho = eho.EHO()
    new_eho.compile(search_space)

    centers = [np.random.normal(size=(2, 1)) for _ in range(10)]

    new_eho._updating_operator(search_space.agents, centers, square)


def test_eho_separating_operator():
    search_space = search.SearchSpace(
        n_agents=20, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_eho = eho.EHO()
    new_eho.compile(search_space)

    new_eho._separating_operator(search_space.agents)


def test_eho_update():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(
        n_agents=20, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_eho = eho.EHO()
    new_eho.compile(search_space)

    new_eho.update(search_space, square)
