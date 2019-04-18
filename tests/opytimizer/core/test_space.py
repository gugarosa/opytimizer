import pytest
import numpy as np
from opytimizer.core import agent, space


def test_space_n_agents():
    new_space = space.Space(n_agents=1)

    assert new_space.n_agents == 1


def test_space_n_variables():
    new_space = space.Space(n_variables=2)

    assert new_space.n_variables == 2


def test_space_n_dimensions():
    new_space = space.Space(n_dimensions=1)

    assert new_space.n_dimensions == 1


def test_space_n_iterations():
    new_space = space.Space(n_iterations=10)

    assert new_space.n_iterations == 10


def test_space_agents():
    new_space = space.Space()

    assert new_space.agents == None


def test_space_agents_setter():
    new_space = space.Space()

    new_space.agents = []

    assert new_space.agents == []


def test_space_best_agent():
    new_space = space.Space()

    assert new_space.best_agent == None


def test_space_best_agent_setter():
    new_space = space.Space()

    new_space.best_agent = agent.Agent()

    assert type(new_space.best_agent).__name__ == 'Agent'


def test_space_lb():
    new_space = space.Space(n_variables=10)

    assert new_space.lb.shape == (10, )


def test_space_lb_setter():
    new_space = space.Space(n_variables=2)

    new_space.lb = np.array([0, 1])

    assert new_space.lb.shape == (2, )


def test_space_ub():
    new_space = space.Space(n_variables=10)

    assert new_space.ub.shape == (10, )


def test_space_ub_setter():
    new_space = space.Space(n_variables=2)

    new_space.ub = np.array([0, 1])

    assert new_space.ub.shape == (2, )


def test_space_check_bound_size():
    new_space = space.Space()

    lb = [0, 1, 2, 3, 4]

    size = 5

    try:
        new_space._check_bound_size(lb, size-1)
    except:
        boolean = new_space._check_bound_size(lb, size)

    assert boolean == True


def test_space_create_agents():
    new_space = space.Space(n_agents=2, n_variables=2, n_dimensions=1)

    new_space.agents, new_space.best_agent = new_space._create_agents(2, 2, 1)

    assert len(new_space.agents) == 2


def test_space_initialize_agents():
    new_space = space.Space(n_agents=2, n_variables=2, n_dimensions=1)

    with pytest.raises(NotImplementedError):
        new_space._initialize_agents(
            new_space.agents, new_space.lb, new_space.ub)


def test_space_build():
    new_space = space.Space()

    try:
        lb = None

        ub = [10, 10]

        new_space._build(lb, ub)
    except:
        lb = [0, 0]

        ub = [10, 10]

        new_space._build(lb, ub)

    try:
        lb = [0, 0]

        ub = None

        new_space._build(lb, ub)
    except:
        lb = [0, 0]

        ub = [10, 10]

        new_space._build(lb, ub)

    assert new_space.built == True
