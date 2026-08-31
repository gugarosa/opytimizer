import numpy as np

from opytimizer.optimizers.evolutionary import ep
from opytimizer.spaces import search


def test_ep_params():
    params = {"bout_size": 0.1, "clip_ratio": 0.05}

    new_ep = ep.EP(params=params)

    assert new_ep.bout_size == 0.1

    assert new_ep.clip_ratio == 0.05


def test_ep_compile():
    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_ep = ep.EP()
    new_ep.compile(search_space)


def test_ep_mutate_parent():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_ep = ep.EP()
    new_ep.compile(search_space)

    agent = new_ep._mutate_parent(search_space.agents[0], 0, square)

    assert agent.position[0][0] > 0


def test_ep_update_strategy():
    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_ep = ep.EP()
    new_ep.compile(search_space)

    new_ep._update_strategy(0, [1], [2])

    assert new_ep.strategy[0][0] > 0


def test_ep_update():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_ep = ep.EP()
    new_ep.compile(search_space)

    new_ep.update(search_space, square)
