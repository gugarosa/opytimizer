import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.science import bh
from opytimizer.spaces import search


def test_bh_update_position():
    def square(x):
        return np.sum(x**2)

    new_function = function.Function(pointer=square)

    new_bh = bh.BH()

    search_space = search.SearchSpace(
        n_agents=20, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    cost = new_bh._update_position(
        search_space.agents, search_space.best_agent, new_function
    )

    assert cost != 0


def test_bh_event_horizon():
    new_bh = bh.BH()

    search_space = search.SearchSpace(
        n_agents=20, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_bh._event_horizon(search_space.agents, search_space.best_agent, 10)

    assert search_space.best_agent.fit != 0


def test_bh_update():
    def square(x):
        return np.sum(x**2)

    new_bh = bh.BH()

    search_space = search.SearchSpace(
        n_agents=20, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_bh.update(search_space, square)
