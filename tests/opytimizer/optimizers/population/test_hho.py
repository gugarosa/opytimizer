import numpy as np

from opytimizer.optimizers.population import hho
from opytimizer.spaces import search

np.random.seed(0)


def test_hho_calculate_initial_coefficients():
    new_hho = hho.HHO()

    E, J = new_hho._calculate_initial_coefficients(1, 10)

    assert E[0] != 0
    assert J[0] != 0


def test_hho_exploration_phase():
    new_hho = hho.HHO()

    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_hho._exploration_phase(
        search_space.agents, search_space.agents[0], search_space.best_agent)


def test_hho_exploitation_phase():
    def square(x):
        return np.sum(x**2)

    new_hho = hho.HHO()

    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_hho._exploitation_phase(
        1, 1, search_space.agents, search_space.agents[0], search_space.best_agent, square)


def test_hho_update():
    def square(x):
        return np.sum(x**2)

    new_hho = hho.HHO()

    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_hho.update(search_space, square, 1, 10)
