import numpy as np
import pytest

from opytimizer.core import agent


def test_agent_creation():
    new_agent = agent.Agent(n_variables=5, n_dimensions=4)
    assert new_agent.position.shape == (5, 4)

def test_check_limits():
    new_agent = agent.Agent(n_variables=5, n_dimensions=4)
    lower_bound = np.ones(new_agent.n_variables)
    upper_bound = 2 * np.ones(new_agent.n_variables)
    new_agent.check_limits(lower_bound, upper_bound)
    assert np.all(new_agent.position == 1)
