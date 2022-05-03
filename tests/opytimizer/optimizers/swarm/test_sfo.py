import numpy as np

from opytimizer.optimizers.swarm import sfo
from opytimizer.spaces import search

np.random.seed(0)


def test_sfo_params():
    params = {"PP": 0.1, "A": 4, "e": 0.001}

    new_sfo = sfo.SFO(params=params)

    assert new_sfo.PP == 0.1

    assert new_sfo.A == 4

    assert new_sfo.e == 0.001


def test_sfo_params_setter():
    new_sfo = sfo.SFO()

    try:
        new_sfo.PP = "a"
    except:
        new_sfo.PP = 0.1

    try:
        new_sfo.PP = -1
    except:
        new_sfo.PP = 0.1

    assert new_sfo.PP == 0.1

    try:
        new_sfo.A = "b"
    except:
        new_sfo.A = 4

    try:
        new_sfo.A = 0
    except:
        new_sfo.A = 4

    assert new_sfo.A == 4

    try:
        new_sfo.e = "c"
    except:
        new_sfo.e = 0.001

    try:
        new_sfo.e = -1
    except:
        new_sfo.e = 0.001

    assert new_sfo.e == 0.001


def test_sfo_compile():
    search_space = search.SearchSpace(
        n_agents=10, n_variables=3, lower_bound=[0, 0, 0], upper_bound=[10, 10, 10]
    )

    new_sfo = sfo.SFO()
    new_sfo.compile(search_space)

    try:
        new_sfo.sardines = 1
    except:
        new_sfo.sardines = []

    assert new_sfo.sardines == []


def test_sfo_generate_random_agent():
    search_space = search.SearchSpace(
        n_agents=10, n_variables=3, lower_bound=[0, 0, 0], upper_bound=[10, 10, 10]
    )

    new_sfo = sfo.SFO()
    new_sfo.compile(search_space)

    agent = new_sfo._generate_random_agent(search_space.agents[0])

    assert type(agent).__name__ == "Agent"


def test_sfo_calculate_lambda_i():
    search_space = search.SearchSpace(
        n_agents=10, n_variables=3, lower_bound=[0, 0, 0], upper_bound=[10, 10, 10]
    )

    new_sfo = sfo.SFO()
    new_sfo.compile(search_space)

    lambda_i = new_sfo._calculate_lambda_i(10, 10)

    assert lambda_i[0] != 0


def test_sfo_update_sailfish():
    search_space = search.SearchSpace(
        n_agents=10, n_variables=3, lower_bound=[0, 0, 0], upper_bound=[10, 10, 10]
    )

    new_sfo = sfo.SFO()
    new_sfo.compile(search_space)

    position = new_sfo._update_sailfish(
        search_space.agents[0], search_space.best_agent, search_space.agents[0], 0.5
    )

    assert position[0][0] != 0


def test_sfo_update():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(
        n_agents=10,
        n_variables=5,
        lower_bound=[0, 0, 0, 0, 0],
        upper_bound=[10, 10, 10, 10, 10],
    )

    new_sfo = sfo.SFO()
    new_sfo.compile(search_space)

    new_sfo.update(search_space, square, 1)

    new_sfo.A = 1
    new_sfo.update(search_space, square, 350)
