import numpy as np

from opytimizer.core import function
from opytimizer.optimizers import bha
from opytimizer.spaces import search
from opytimizer.utils import constants


def test_bha_build():
    new_bha = bha.BHA()

    assert new_bha.built == True


def test_bha_update_position():
    def square(x):
        return np.sum(x**2)

    new_function = function.Function(pointer=square)

    new_bha = bha.BHA()

    search_space = search.SearchSpace(n_agents=20, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    cost = new_bha._update_position(
        search_space.agents, search_space.best_agent, new_function)

    assert cost != 0


def test_bha_event_horizon():
    new_bha = bha.BHA()

    search_space = search.SearchSpace(n_agents=20, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    new_bha._event_horizon(search_space.agents, search_space.best_agent, 10)

    assert search_space.best_agent.fit != 0


def test_bha_run():
    def square(x):
        return np.sum(x)

    new_function = function.Function(pointer=square)

    new_bha = bha.BHA()

    search_space = search.SearchSpace(n_agents=10, n_iterations=50,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[5, 5])

    history = new_bha.run(search_space, new_function)

    assert len(history.agents) > 0
    assert len(history.best) > 0
    assert len(history.best_index) > 0

    best_fitness = history.best[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, 'The algorithm bha failed to converge.'
