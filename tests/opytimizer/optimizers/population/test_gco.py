import numpy as np

from opytimizer.optimizers.population import gco
from opytimizer.spaces import search

np.random.seed(0)


def test_gco_params():
    params = {"CR": 0.7, "F": 1.25}

    new_gco = gco.GCO(params=params)

    assert new_gco.CR == 0.7

    assert new_gco.F == 1.25


def test_gco_params_setter():
    new_gco = gco.GCO()

    try:
        new_gco.CR = "a"
    except:
        new_gco.CR = 0.75

    try:
        new_gco.CR = -1
    except:
        new_gco.CR = 0.75

    assert new_gco.CR == 0.75

    try:
        new_gco.F = "b"
    except:
        new_gco.F = 1.5

    try:
        new_gco.F = -1
    except:
        new_gco.F = 1.5

    assert new_gco.F == 1.5


def test_gco_compile():
    search_space = search.SearchSpace(
        n_agents=4, n_variables=2, lower_bound=[1, 1], upper_bound=[10, 10]
    )

    new_gco = gco.GCO()
    new_gco.compile(search_space)

    try:
        new_gco.life = 1
    except:
        new_gco.life = np.array([1])

    assert new_gco.life == np.array([1])

    try:
        new_gco.counter = 1
    except:
        new_gco.counter = np.array([1])

    assert new_gco.counter == np.array([1])


def test_gco_mutate_cell():
    search_space = search.SearchSpace(
        n_agents=4, n_variables=2, lower_bound=[1, 1], upper_bound=[10, 10]
    )

    new_gco = gco.GCO()
    new_gco.compile(search_space)

    cell = new_gco._mutate_cell(
        search_space.agents[0],
        search_space.agents[1],
        search_space.agents[2],
        search_space.agents[3],
    )

    assert cell.position[0][0] != 0


def test_gco_dark_zone():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(
        n_agents=4, n_variables=2, lower_bound=[1, 1], upper_bound=[10, 10]
    )

    new_gco = gco.GCO()
    new_gco.compile(search_space)

    new_gco._dark_zone(search_space.agents, square)


def test_gco_light_zone():
    search_space = search.SearchSpace(
        n_agents=4, n_variables=2, lower_bound=[1, 1], upper_bound=[10, 10]
    )

    new_gco = gco.GCO()
    new_gco.compile(search_space)

    new_gco._light_zone(search_space.agents)


def test_gco_update():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(
        n_agents=4, n_variables=2, lower_bound=[1, 1], upper_bound=[10, 10]
    )

    new_gco = gco.GCO()
    new_gco.compile(search_space)

    new_gco.update(search_space, square)

    assert search_space.agents[0].position[0] != 0
