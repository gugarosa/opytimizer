import pytest

from opytimizer.spaces import search


def test_search_initialize_agents():
    lb = [0, 0]

    ub = [10, 10]

    new_search_space = search.SearchSpace(lower_bound=lb, upper_bound=ub)

    assert new_search_space.agents[0].position[0] != 0


def test_search_check_bound_limits():
    lb = [0, 0]

    ub = [10, 10]

    new_search_space = search.SearchSpace(lower_bound=lb, upper_bound=ub)

    new_search_space.agents[0].position[0] = 20

    new_search_space.check_bound_limits(new_search_space.agents, lb, ub)

    assert new_search_space.agents[0].position[0] != 20
