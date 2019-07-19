import sys

import numpy as np
import pytest

from opytimizer.core import agent


def test_agent_n_variables():
    new_agent = agent.Agent(n_variables=5, n_dimensions=4)

    assert new_agent.n_variables == 5


def test_agent_n_dimensions():
    new_agent = agent.Agent(n_variables=5, n_dimensions=4)

    assert new_agent.n_dimensions == 4


def test_agent_position():
    new_agent = agent.Agent(n_variables=5, n_dimensions=4)

    assert new_agent.position.shape == (5, 4)


def test_agent_position_setter():
    new_agent = agent.Agent(n_variables=1, n_dimensions=1)

    new_agent.position = 10

    assert new_agent.position == 10


def test_agent_fit():
    new_agent = agent.Agent(n_variables=5, n_dimensions=4)

    assert new_agent.fit == sys.float_info.max


def test_agent_fit_setter():
    new_agent = agent.Agent(n_variables=5, n_dimensions=4)

    new_agent.fit = 0

    assert new_agent.fit == 0


def test_agent_check_limits():
    new_agent = agent.Agent(n_variables=1, n_dimensions=1)

    new_agent.lb = [10]
    
    new_agent.ub = [10]

    new_agent.check_limits()

    assert new_agent.position[0] == 10
