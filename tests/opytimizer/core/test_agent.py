# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

import sys
from typing import get_type_hints

import numpy as np
import pytest
from numpy.typing import ArrayLike

from opytimizer.core import Agent


def test_agent_initializes_state_and_mapping():
    agent = Agent(2, 3, [-1, -2], [1, 2], ["a", "b"])

    assert agent.position.shape == (2, 3)
    assert agent.fit == sys.float_info.max
    assert agent.mapping == ["a", "b"]
    assert set(agent.mapped_position) == {"a", "b"}
    assert isinstance(agent.ts, int)


@pytest.mark.parametrize(
    "args,error",
    [
        ((1.0, 1, 0, 1), TypeError),
        ((0, 1, 0, 1), ValueError),
        ((1, 1.0, 0, 1), TypeError),
        ((1, 0, 0, 1), ValueError),
        ((2, 1, [0], [1, 1]), ValueError),
        ((2, 1, [0, 0], [1]), ValueError),
        ((1, 1, 0, 1, "x"), TypeError),
        ((2, 1, [0, 0], [1, 1], ["x"]), ValueError),
    ],
)
def test_agent_validates_constructor_inputs(args, error):
    with pytest.raises(error):
        Agent(*args)


def test_agent_clips_and_fills_positions():
    agent = Agent(2, 2, [0, -2], [1, -1])

    agent.position[:] = 10
    agent.clip_by_bound()
    assert np.array_equal(agent.position, [[1, 1], [-1, -1]])

    agent.fill_with_binary()
    assert set(np.unique(agent.position)) <= {0, 1}

    agent.fill_with_static([0.5, -1.5])
    assert np.array_equal(agent.position, [[0.5, 0.5], [-1.5, -1.5]])

    agent.fill_with_uniform()
    assert np.all(agent.position[0] >= 0)
    assert np.all(agent.position[0] <= 1)
    assert np.all(agent.position[1] >= -2)
    assert np.all(agent.position[1] <= -1)


def test_agent_rejects_static_values_with_wrong_size():
    with pytest.raises(ValueError):
        Agent(2, 1, [0, 0], [1, 1]).fill_with_static([1])


@pytest.mark.parametrize("values", [0.5, (0.5,), [0.5], np.array([0.5]), np.array([[0.5, 0.5]])])
def test_agent_accepts_array_like_bounds_and_static_values(values):
    agent = Agent(1, 2, values, values)

    agent.fill_with_static(values)

    assert np.array_equal(agent.position, [[0.5, 0.5]])
    assert np.array_equal(agent.lb, np.atleast_1d(values))
    assert np.array_equal(agent.ub, np.atleast_1d(values))


def test_agent_annotations_accept_array_like_inputs_and_nullable_bounds():
    constructor_hints = get_type_hints(Agent.__init__)
    fill_hints = get_type_hints(Agent.fill_with_static)

    assert constructor_hints["lower_bound"] == ArrayLike | None
    assert constructor_hints["upper_bound"] == ArrayLike | None
    assert fill_hints["values"] == ArrayLike
