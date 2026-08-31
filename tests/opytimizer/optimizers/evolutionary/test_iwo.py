import numpy as np

from opytimizer.optimizers.evolutionary import iwo
from opytimizer.spaces import search

np.random.seed(0)


def test_iwo_params():
    params = {
        "min_seeds": 0,
        "max_seeds": 5,
        "e": 2,
        "final_sigma": 0.001,
        "init_sigma": 3,
    }

    new_iwo = iwo.IWO(params=params)

    assert new_iwo.min_seeds == 0

    assert new_iwo.max_seeds == 5

    assert new_iwo.e == 2

    assert new_iwo.final_sigma == 0.001

    assert new_iwo.init_sigma == 3


def test_iwo_spatial_dispersal():
    new_iwo = iwo.IWO()

    new_iwo._spatial_dispersal(1, 10)

    assert new_iwo.sigma == 2.43019


def test_iwo_produce_offspring():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(
        n_agents=2, n_variables=2, lower_bound=[1, 1], upper_bound=[10, 10]
    )

    new_iwo = iwo.IWO()

    agent = new_iwo._produce_offspring(search_space.agents[0], square)

    assert type(agent).__name__ == "Agent"


def test_iwo_update():
    def square(x):
        return np.sum(x**2)

    new_iwo = iwo.IWO()
    new_iwo.min_seeds = 5
    new_iwo.max_seeds = 20

    search_space = search.SearchSpace(
        n_agents=5, n_variables=2, lower_bound=[1, 1], upper_bound=[10, 10]
    )

    new_iwo.update(search_space, square, 1, 10)
