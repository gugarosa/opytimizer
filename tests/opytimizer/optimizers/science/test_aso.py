import numpy as np

from opytimizer.optimizers.science import aso
from opytimizer.spaces import search


def test_aso_params():
    params = {
        'alpha': 50.0,
        'beta': 0.2
    }

    new_aso = aso.ASO(params=params)

    assert new_aso.alpha == 50.0

    assert new_aso.beta == 0.2


def test_aso_params_setter():
    new_aso = aso.ASO()

    try:
        new_aso.alpha = 'a'
    except:
        new_aso.alpha = 50.0

    try:
        new_aso.beta = 'b'
    except:
        new_aso.beta = 0.2

    try:
        new_aso.beta = -1
    except:
        new_aso.beta = 0.2

    assert new_aso.beta == 0.2


def test_aso_compile():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_aso = aso.ASO()
    new_aso.compile(search_space)

    try:
        new_aso.velocity = 1
    except:
        new_aso.velocity = np.array([1])

    assert new_aso.velocity == np.array([1])


def test_aso_calculate_mass():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_aso = aso.ASO()
    new_aso.compile(search_space)

    mass = new_aso._calculate_mass(search_space.agents)

    assert mass[0] == 0.1


def test_aso_calculate_potential():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_aso = aso.ASO()
    new_aso.compile(search_space)

    new_aso._calculate_potential(
        search_space.agents[0], search_space.agents[1], np.array([1]), 1, 10)


def test_aso_calculate_acceleration():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_aso = aso.ASO()
    new_aso.compile(search_space)

    mass = new_aso._calculate_mass(search_space.agents)
    new_aso._calculate_acceleration(
        search_space.agents, search_space.best_agent, mass, 1, 10)


def test_aso_update():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_aso = aso.ASO()
    new_aso.compile(search_space)

    new_aso.update(search_space, 1, 10)
