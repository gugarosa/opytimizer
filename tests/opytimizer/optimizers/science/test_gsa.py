import numpy as np

from opytimizer.optimizers.science import gsa
from opytimizer.spaces import search


def test_gsa_params():
    params = {
        "G": 2.467,
    }

    new_gsa = gsa.GSA(params=params)

    assert new_gsa.G == 2.467


def test_gsa_params_setter():
    new_gsa = gsa.GSA()

    try:
        new_gsa.G = "a"
    except:
        new_gsa.G = 0.1

    try:
        new_gsa.G = -1
    except:
        new_gsa.G = 0.1

    assert new_gsa.G == 0.1


def test_gsa_compile():
    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_gsa = gsa.GSA()
    new_gsa.compile(search_space)

    try:
        new_gsa.velocity = 1
    except:
        new_gsa.velocity = np.array([1])

    assert new_gsa.velocity == np.array([1])


def test_gsa_calculate_mass():
    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_gsa = gsa.GSA()
    new_gsa.compile(search_space)

    search_space.agents[0].fit = 1

    search_space.agents.sort(key=lambda x: x.fit)

    mass = new_gsa._calculate_mass(search_space.agents)

    assert len(mass) > 0


def test_gsa_calculate_force():
    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_gsa = gsa.GSA()
    new_gsa.compile(search_space)

    search_space.agents[0].fit = 1

    search_space.agents.sort(key=lambda x: x.fit)

    mass = new_gsa._calculate_mass(search_space.agents)

    gravity = 1

    force = new_gsa._calculate_force(search_space.agents, mass, gravity)

    assert force.shape[0] > 0


def test_gsa_update():
    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_gsa = gsa.GSA()
    new_gsa.compile(search_space)

    new_gsa.update(search_space, 1)
