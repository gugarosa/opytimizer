import numpy as np

from opytimizer.optimizers.population import rfo
from opytimizer.spaces import search

np.random.seed(1)


def test_rfo_params():
    params = {
        "phi": np.random.uniform(0, 2 * np.pi),
        "theta": np.random.uniform(),
        "p_replacement": 0.05,
    }

    new_rfo = rfo.RFO(params=params)

    assert 0 <= new_rfo.phi <= 2 * np.pi

    assert 0 <= new_rfo.theta <= 1

    assert new_rfo.p_replacement == 0.05


def test_rfo_compile():
    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_rfo = rfo.RFO()
    new_rfo.compile(search_space)


def test_rfo_rellocation():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_rfo = rfo.RFO()
    new_rfo.compile(search_space)

    new_rfo._rellocation(search_space.agents[0], search_space.best_agent, square)


def test_rfo_noticing():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_rfo = rfo.RFO()
    new_rfo.compile(search_space)

    new_rfo._noticing(search_space.agents[0], square, 0.1)

    new_rfo.phi = 0
    new_rfo._noticing(search_space.agents[0], square, 0.1)


def test_rfo_update():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_rfo = rfo.RFO()
    new_rfo.compile(search_space)

    new_rfo.update(search_space, square)

    new_rfo.n_replacement = 10
    new_rfo.update(search_space, square)
