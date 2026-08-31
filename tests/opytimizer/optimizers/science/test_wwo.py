import numpy as np

from opytimizer.optimizers.science import wwo
from opytimizer.spaces import search

np.random.seed(0)


def test_wwo_params():
    params = {"h_max": 5, "alpha": 1.001, "beta": 0.001, "k_max": 1}

    new_wwo = wwo.WWO(params=params)

    assert new_wwo.h_max == 5

    assert new_wwo.alpha == 1.001

    assert new_wwo.beta == 0.001

    assert new_wwo.k_max == 1


def test_wwo_compile():
    search_space = search.SearchSpace(
        n_agents=50, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_wwo = wwo.WWO()
    new_wwo.compile(search_space)


def test_wwo_propagate_wave():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(
        n_agents=50, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_wwo = wwo.WWO()
    new_wwo.compile(search_space)

    wave = new_wwo._propagate_wave(search_space.agents[0], square, 0)

    assert type(wave).__name__ == "Agent"


def test_wwo_refract_wave():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(
        n_agents=50, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_wwo = wwo.WWO()
    new_wwo.compile(search_space)

    height, length = new_wwo._refract_wave(
        search_space.agents[0], search_space.best_agent, square, 0
    )

    assert height == 5
    assert length != 0


def test_wwo_break_wave():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(
        n_agents=50, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_wwo = wwo.WWO()
    new_wwo.compile(search_space)

    broken_wave = new_wwo._break_wave(search_space.agents[0], square, 0)

    assert type(broken_wave).__name__ == "Agent"


def test_wwo_update_wave_length():
    search_space = search.SearchSpace(
        n_agents=50, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_wwo = wwo.WWO()
    new_wwo.compile(search_space)

    new_wwo._update_wave_length(search_space.agents)


def test_wwo_update():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(
        n_agents=50, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_wwo = wwo.WWO()
    new_wwo.compile(search_space)

    new_wwo.update(search_space, square)
    new_wwo.update(search_space, square)
