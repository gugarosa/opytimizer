import pytest

from opytimizer.core import space


def test_space_n_agents():
    lb = [-1, -1]
    ub = [1, 1]

    s = space.Space(lower_bound=lb, upper_bound=ub)

    assert s.n_agents == 1


def test_space_n_variables():
    lb = [-1, -1]
    ub = [1, 1]

    s = space.Space(lower_bound=lb, upper_bound=ub)

    assert s.n_variables == 2


def test_space_n_dimensions():
    lb = [-1, -1]
    ub = [1, 1]

    s = space.Space(lower_bound=lb, upper_bound=ub)

    assert s.n_dimensions == 1


def test_space_n_iterations():
    lb = [-1, -1]
    ub = [1, 1]

    s = space.Space(lower_bound=lb, upper_bound=ub)

    assert s.n_iterations == 10
