import pytest

from opytimizer.core import agent
from opytimizer.utils import common


def test_check_bound_limits():
    n_agents = 1
    n_variables = 2
    n_dimensions = 1

    agents = []

    lb = [3, 3]
    ub = [5, 5]

    for _ in range(n_agents):
        agents.append(agent.Agent(n_variables=n_variables,
                                  n_dimensions=n_dimensions))

    assert agents[0].position[0] == 0

    common.check_bound_limits(agents, lb, ub)

    assert agents[0].position[0] == 3
