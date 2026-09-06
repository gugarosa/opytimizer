# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

import numpy as np
import pytest

from opytimizer import Opytimizer
from opytimizer.optimizers.misc import GS
from opytimizer.spaces import GridSpace


def test_grid_space_builds_grid_and_agents_synchronously():
    space = GridSpace(1, 0.1, 0, 1)

    assert np.array_equal(space.step, [0.1])
    assert len(space.grid) == 11
    assert len(space.agents) == 11
    assert np.array_equal(space.agents[0].position, [[0]])
    assert np.allclose(space.agents[-1].position, [[1]])
    assert not hasattr(space, "built")


def test_grid_space_validates_step_size():
    with pytest.raises(ValueError):
        GridSpace(2, [0.1], [0, 0], [1, 1])


@pytest.mark.parametrize(
    "step,lower,upper,expected",
    [
        (2, 0, 5, [0, 2, 4]),
        (2, 0, 4, [0, 2, 4]),
        (2, 1, 1, [1]),
        (3, 1, 2, [1]),
        (0.1, 0, 0.3, [0, 0.1, 0.2, 0.3]),
        (0.1, -0.3, 0, [-0.3, -0.2, -0.1, 0]),
        (0.1, -1, -0.9, [-1, -0.9]),
        (0.2, -0.3, 0.25, [-0.3, -0.1, 0.1]),
        (
            3,
            10_000_000_000_000_000,
            10_000_000_000_000_004,
            [1e16, 1e16 + 4],
        ),
        (
            4_000_000_000_000_000_000,
            -8_000_000_000_000_000_000,
            8_000_000_000_000_000_000,
            [-8e18, -4e18, 0, 4e18, 8e18],
        ),
    ],
)
def test_grid_space_uses_bounded_step_lattice(step, lower, upper, expected):
    space = GridSpace(1, step, lower, upper)

    np.testing.assert_allclose(space.grid[:, 0], expected, rtol=0, atol=1e-15)
    assert np.all(space.grid >= lower)
    assert np.all(space.grid <= upper)
    assert space.n_agents == len(expected)
    np.testing.assert_array_equal([agent.position[:, 0] for agent in space.agents], space.grid)


def test_grid_space_preserves_cartesian_order():
    space = GridSpace(2, [2, 1], [0, 10], [5, 11])

    np.testing.assert_array_equal(space.grid, [[0, 10], [2, 10], [4, 10], [0, 11], [2, 11], [4, 11]])


@pytest.mark.parametrize("step", [0, -1, np.nan, np.inf])
def test_grid_space_rejects_invalid_steps(step):
    with pytest.raises(ValueError, match="`step`"):
        GridSpace(1, step, 0, 1)


@pytest.mark.parametrize("lower,upper", [(2, 1), (np.nan, 1), (0, np.inf), (-np.inf, 0)])
def test_grid_space_rejects_invalid_bounds(lower, upper):
    with pytest.raises(ValueError, match="`lower_bound` and `upper_bound`"):
        GridSpace(1, 1, lower, upper)


def test_grid_search_never_evaluates_or_retains_out_of_bounds_positions():
    evaluated = []

    def objective(position):
        value = float(position[0, 0])
        evaluated.append(value)
        return -value

    model = Opytimizer(GridSpace(1, 2, 0, 5), GS(), objective)
    model.start()

    assert evaluated
    assert set(evaluated) == {0, 2, 4}
    assert model.space.best_agent.position[0, 0] == 4
    assert model.space.best_agent.fit == -4
