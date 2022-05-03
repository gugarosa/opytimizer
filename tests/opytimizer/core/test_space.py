import numpy as np
import pytest

from opytimizer.core import agent, space


def test_space_n_agents():
    new_space = space.Space(n_agents=1)

    assert new_space.n_agents == 1


def test_space_n_agents_setter():
    try:
        new_space = space.Space(n_agents=0.0)
    except:
        new_space = space.Space(n_agents=1)

    try:
        new_space = space.Space(n_agents=0)
    except:
        new_space = space.Space(n_agents=1)

    assert new_space.n_agents == 1


def test_space_n_variables():
    new_space = space.Space(n_variables=1)

    assert new_space.n_variables == 1


def test_space_n_variables_setter():
    try:
        new_space = space.Space(n_variables=0.0)
    except:
        new_space = space.Space(n_variables=1)

    try:
        new_space = space.Space(n_variables=0)
    except:
        new_space = space.Space(n_variables=1)

    assert new_space.n_variables == 1


def test_space_n_dimensions():
    new_space = space.Space(n_dimensions=1)

    assert new_space.n_dimensions == 1


def test_space_n_dimensions_setter():
    try:
        new_space = space.Space(n_dimensions=0.0)
    except:
        new_space = space.Space(n_dimensions=1)

    try:
        new_space = space.Space(n_dimensions=0)
    except:
        new_space = space.Space(n_dimensions=1)

    assert new_space.n_dimensions == 1


def test_space_agents():
    new_space = space.Space()

    assert new_space.agents == []


def test_space_agents_setter():
    new_space = space.Space()

    try:
        new_space.agents = None
    except:
        new_space.agents = []

    assert new_space.agents == []


def test_space_best_agent():
    new_space = space.Space()

    assert isinstance(new_space.best_agent, agent.Agent)


def test_space_best_agent_setter():
    new_space = space.Space()

    try:
        new_space.best_agent = None
    except:
        new_space.best_agent = agent.Agent(1, 1, 0, 1)

    assert isinstance(new_space.best_agent, agent.Agent)


def test_space_lb():
    new_space = space.Space(n_variables=1)

    assert new_space.lb.shape == (1,)


def test_space_lb_setter():
    new_space = space.Space(n_variables=1)

    try:
        new_space.lb = [1]
    except:
        new_space.lb = np.array([1])

    assert new_space.lb[0] == 1

    try:
        new_space.lb = np.array([1, 2])
    except:
        new_space.lb = np.array([1])

    assert new_space.lb[0] == 1


def test_space_ub():
    new_space = space.Space(n_variables=1)

    assert new_space.ub.shape == (1,)


def test_space_ub_setter():
    new_space = space.Space(n_variables=1)

    try:
        new_space.ub = [1]
    except:
        new_space.ub = np.array([1])

    assert new_space.ub[0] == 1

    try:
        new_space.ub = np.array([1, 2])
    except:
        new_space.ub = np.array([1])

    assert new_space.ub[0] == 1


def test_space_built():
    new_space = space.Space()

    assert new_space.built is False


def test_space_built_setter():
    new_space = space.Space()

    try:
        new_space.built = 1
    except:
        new_space.built = True

    assert new_space.built is True


def test_space_create_agents():
    new_space = space.Space(n_agents=2, n_variables=1, n_dimensions=1)

    new_space._create_agents()

    assert len(new_space.agents) == 2


def test_space_initialize_agents():
    new_space = space.Space(n_agents=2, n_variables=1, n_dimensions=1)

    new_space._initialize_agents()


def test_space_build():
    new_space = space.Space()

    new_space.build()

    assert new_space.built is True


def test_space_clip_by_bound():
    new_space = space.Space()

    new_space.build()
    new_space.clip_by_bound()

    assert new_space.agents[0].position[0] == 0
