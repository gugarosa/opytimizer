import sys

import numpy as np
import pytest

from opytimizer.core import agent


def test_agent_n_variables():
    new_agent = agent.Agent(n_variables=5, n_dimensions=4)

    assert new_agent.n_variables == 5


def test_agent_n_variables_setter():
    try:
        new_agent = agent.Agent(n_variables=0.0, n_dimensions=4)
    except:
        new_agent = agent.Agent(n_variables=5, n_dimensions=4)

    try:
        new_agent = agent.Agent(n_variables=0, n_dimensions=4)
    except:
        new_agent = agent.Agent(n_variables=5, n_dimensions=4)

    assert new_agent.n_variables == 5


def test_agent_n_dimensions():
    new_agent = agent.Agent(n_variables=5, n_dimensions=4)

    assert new_agent.n_dimensions == 4


def test_agent_n_dimensions_setter():
    try:
        new_agent = agent.Agent(n_variables=5, n_dimensions=0.0)
    except:
        new_agent = agent.Agent(n_variables=5, n_dimensions=4)

    try:
        new_agent = agent.Agent(n_variables=5, n_dimensions=0)
    except:
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


def test_agent_lb():
    new_agent = agent.Agent(n_variables=5, n_dimensions=4)

    assert len(new_agent.lb) == 5


def test_agent_lb_setter():
    new_agent = agent.Agent(n_variables=5, n_dimensions=4)

    new_agent.lb[0] = 1

    assert new_agent.lb[0] == 1


def test_agent_ub():
    new_agent = agent.Agent(n_variables=5, n_dimensions=4)

    assert len(new_agent.ub) == 5


def test_agent_ub_setter():
    new_agent = agent.Agent(n_variables=5, n_dimensions=4)

    new_agent.ub[0] = 1

    assert new_agent.ub[0] == 1


def test_agent_clip_limits():
    new_agent = agent.Agent(n_variables=1, n_dimensions=1)

    new_agent.lb = [10]

    new_agent.ub = [10]

    new_agent.clip_limits()

    assert new_agent.position[0] == 10
