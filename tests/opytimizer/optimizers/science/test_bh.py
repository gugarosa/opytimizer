import numpy as np

from opytimizer.core import function
from opytimizer.optimizers import bh
from opytimizer.spaces import search
from opytimizer.utils import constants


def test_bh_build():
    new_bh = bh.BH()

    assert new_bh.built == True


def test_bh_update_position():
    def square(x):
        return np.sum(x**2)

    new_function = function.Function(pointer=square)

    new_bh = bh.BH()

    search_space = search.SearchSpace(n_agents=20, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    cost = new_bh._update_position(
        search_space.agents, search_space.best_agent, new_function)

    assert cost != 0


def test_bh_event_horizon():
    new_bh = bh.BH()

    search_space = search.SearchSpace(n_agents=20, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    new_bh._event_horizon(search_space.agents, search_space.best_agent, 10)

    assert search_space.best_agent.fit != 0


def test_bh_run():
    def square(x):
        return np.sum(x)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    new_bh = bh.BH()

    search_space = search.SearchSpace(n_agents=10, n_iterations=50,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[5, 5])

    history = new_bh.run(search_space, new_function, pre_evaluation_hook=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, 'The algorithm bh failed to converge.'
