# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

import numpy as np
import pytest

from opytimizer.optimizers.boolean import umda
from opytimizer.spaces import boolean


def test_umda_params():
    params = {"p_selection": 0.75, "lower_bound": 0.05, "upper_bound": 0.95}

    new_umda = umda.UMDA(params=params)

    assert new_umda.p_selection == 0.75

    assert new_umda.lower_bound == 0.05

    assert new_umda.upper_bound == 0.95


def test_umda_calculate_probability():
    new_umda = umda.UMDA()

    boolean_space = boolean.BooleanSpace(n_agents=5, n_variables=2)

    probs = new_umda._calculate_probability(boolean_space.agents)

    assert probs.shape == (2, 1)


def test_umda_sample_position():
    new_umda = umda.UMDA()

    probs = np.zeros((1, 1))

    position = new_umda._sample_position(probs)

    assert position == 1


def test_umda_update():
    new_umda = umda.UMDA()

    boolean_space = boolean.BooleanSpace(n_agents=2, n_variables=5)

    new_umda.update(boolean_space)


@pytest.mark.parametrize(
    "lower_bound, upper_bound",
    [
        (-0.1, 0.95),
        (0.05, -0.1),
        (0.8, 0.2),
        (0.05, 1.1),
        (np.nan, 0.95),
        (0.05, np.nan),
        (-np.inf, 0.95),
        (0.05, np.inf),
    ],
)
def test_umda_rejects_invalid_probability_bounds(lower_bound, upper_bound):
    with pytest.raises(ValueError):
        umda.UMDA({"lower_bound": lower_bound, "upper_bound": upper_bound})


@pytest.mark.parametrize("name", ["lower_bound", "upper_bound"])
@pytest.mark.parametrize("bound", ["0.5", None, 0.5j, np.array([0.5])])
def test_umda_rejects_nonreal_probability_bounds(name, bound):
    with pytest.raises(TypeError, match=rf"`{name}`.*\.$"):
        umda.UMDA({name: bound})


@pytest.mark.parametrize("method", ["update", "_calculate_probability"])
@pytest.mark.parametrize("name, bound", [("upper_bound", -0.1), ("lower_bound", 1.0), ("upper_bound", np.nan)])
def test_umda_reassigned_bounds_fail_before_population_mutation(method, name, bound):
    optimizer = umda.UMDA()
    space = boolean.BooleanSpace(n_agents=3, n_variables=2)
    for index, agent in enumerate(space.agents):
        agent.fit = 3 - index
    agents = list(space.agents)
    positions = [agent.position.copy() for agent in agents]
    setattr(optimizer, name, bound)

    with pytest.raises(ValueError):
        getattr(optimizer, method)(space if method == "update" else space.agents)

    assert all(actual is expected for actual, expected in zip(space.agents, agents))
    for agent, position in zip(space.agents, positions):
        np.testing.assert_array_equal(agent.position, position)


@pytest.mark.parametrize("bound", [np.float32(0.0), np.float64(0.5), np.int64(1)])
def test_umda_accepts_numpy_probability_bounds_and_preserves_sampling_direction(bound, monkeypatch):
    optimizer = umda.UMDA({"lower_bound": bound, "upper_bound": bound})
    space = boolean.BooleanSpace(n_agents=2, n_variables=2)

    probabilities = optimizer._calculate_probability(space.agents)

    np.testing.assert_array_equal(probabilities, np.full((2, 1), bound))
    monkeypatch.setattr(np.random, "uniform", lambda low, high, size: np.full(size, 0.25))
    np.testing.assert_array_equal(optimizer._sample_position(probabilities), np.full((2, 1), bound < 0.25))

    optimizer.update(space)

    for agent in space.agents:
        np.testing.assert_array_equal(agent.position, np.full((2, 1), bound < 0.25))
