import sys

import numpy as np

from opytimizer.core import agent


def test_agent_n_variables():
    new_agent = agent.Agent(1, 1, 0, 1)

    assert new_agent.n_variables == 1


def test_agent_n_variables_setter():
    try:
        new_agent = agent.Agent(0.0, 1, 0, 1)
    except:
        new_agent = agent.Agent(1, 1, 0, 1)

    try:
        new_agent = agent.Agent(0, 4, 0, 1)
    except:
        new_agent = agent.Agent(1, 1, 0, 1)

    assert new_agent.n_variables == 1


def test_agent_n_dimensions():
    new_agent = agent.Agent(1, 1, 0, 1)

    assert new_agent.n_dimensions == 1


def test_agent_n_dimensions_setter():
    try:
        new_agent = agent.Agent(1, 0.0, 0, 1)
    except:
        new_agent = agent.Agent(1, 1, 0, 1)

    try:
        new_agent = agent.Agent(1, 0, 0, 1)
    except:
        new_agent = agent.Agent(1, 1, 0, 1)

    assert new_agent.n_dimensions == 1


def test_agent_position():
    new_agent = agent.Agent(1, 1, 0, 1)

    assert new_agent.position.shape == (1, 1)


def test_agent_position_setter():
    new_agent = agent.Agent(1, 1, 0, 1)

    try:
        new_agent.position = 10
    except:
        new_agent.position = np.array([10])

    assert new_agent.position[0] == 10


def test_agent_fit():
    new_agent = agent.Agent(1, 1, 0, 1)

    assert new_agent.fit == sys.float_info.max


def test_agent_fit_setter():
    new_agent = agent.Agent(1, 1, 0, 1)

    try:
        new_agent.fit = np.array([0])
    except:
        new_agent.fit = 0

    assert new_agent.fit == 0


def test_agent_lb():
    new_agent = agent.Agent(1, 1, 0, 1)

    assert len(new_agent.lb) == 1


def test_agent_lb_setter():
    new_agent = agent.Agent(1, 1, 0, 1)

    try:
        new_agent.lb = [1]
    except:
        new_agent.lb = np.array([1])

    assert new_agent.lb[0] == 1

    try:
        new_agent.lb = np.array([1, 2])
    except:
        new_agent.lb = np.array([1])

    assert new_agent.lb[0] == 1


def test_agent_ub():
    new_agent = agent.Agent(1, 1, 0, 1)

    assert len(new_agent.ub) == 1


def test_agent_ub_setter():
    new_agent = agent.Agent(1, 1, 0, 1)

    try:
        new_agent.ub = [1]
    except:
        new_agent.ub = np.array([1])

    assert new_agent.ub[0] == 1

    try:
        new_agent.ub = np.array([1, 2])
    except:
        new_agent.ub = np.array([1])

    assert new_agent.ub[0] == 1


def test_agent_ts():
    new_agent = agent.Agent(1, 1, 0, 1)

    assert type(new_agent.ts) == int


def test_agent_ts_setter():
    new_agent = agent.Agent(1, 1, 0, 1)

    try:
        new_agent.ts = np.array([0])
    except:
        new_agent.ts = 0

    assert new_agent.ts == 0


def test_agent_mapping():
    new_agent = agent.Agent(1, 1, 0, 1)

    assert len(new_agent.mapping) == 1


def test_agent_mapping_setter():
    new_agent = agent.Agent(1, 1, 0, 1)

    try:
        new_agent.mapping = "a"
    except:
        new_agent.mapping = ["x1"]

    assert len(new_agent.mapping) == 1

    try:
        new_agent.mapping = []
    except:
        new_agent.mapping = ["x1"]

    assert len(new_agent.mapping) == 1


def test_agent_mapped_position():
    new_agent = agent.Agent(1, 1, 0, 1)

    assert new_agent.mapped_position["x0"].shape == (1,)


def test_agent_clip_by_bound():
    new_agent = agent.Agent(1, 1, 0, 1)

    new_agent.lb = np.array([10])

    new_agent.ub = np.array([10])

    new_agent.clip_by_bound()

    assert new_agent.position[0] == 10


def test_agent_fill_with_binary():
    new_agent = agent.Agent(1, 1, 0, 1)

    new_agent.fill_with_binary()

    assert new_agent.position[0] in [0, 1]


def test_agent_fill_with_static():
    new_agent = agent.Agent(1, 1, 0, 1)

    try:
        new_agent.fill_with_static([20, 20])
    except:
        new_agent.fill_with_static(20)

    assert new_agent.position[0] == 20


def test_agent_fill_with_uniform():
    new_agent = agent.Agent(1, 1, 0, 1)

    new_agent.fill_with_uniform()

    assert new_agent.position[0] >= 0
    assert new_agent.position[0] <= 1
