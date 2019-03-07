import pytest
from opytimizer.spaces import hyper


def test_hyper_initialize_agents():
    lb = [0, 0]

    ub = [10, 10]

    new_hyper_space = hyper.HyperSpace(lower_bound=lb, upper_bound=ub)

    assert new_hyper_space.agents[0].position[0][0] > 0


def test_hyper_check_bound_limits():
    lb = [0, 0]

    ub = [10, 10]

    new_hyper_space = hyper.HyperSpace(lower_bound=lb, upper_bound=ub)

    new_hyper_space.agents[0].position[0][0] = 20

    new_hyper_space.check_bound_limits(new_hyper_space.agents, lb, ub)

    assert new_hyper_space.agents[0].position[0][0] != 20
