# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

from typing import get_type_hints

import numpy as np
import pytest

from opytimizer.core import Agent
from opytimizer.utils.history import History


def make_agents():
    return [Agent(2, 1, [0, 0], [1, 1]) for _ in range(2)]


def test_history_validates_save_agents():
    assert History().save_agents is False

    with pytest.raises(TypeError):
        History("yes")


def test_history_dumps_and_parses_values():
    agents = make_agents()
    history = History(save_agents=True)

    history.dump(
        agents=agents,
        best_agent=agents[0],
        local_position=agents[0].position,
        value=1,
    )
    history.dump(
        agents=agents,
        best_agent=agents[0],
        local_position=agents[0].position,
        value=2,
    )

    agents_pos, agents_fit = history.get_convergence("agents", index=0)
    best_pos, best_fit = history.get_convergence("best_agent")

    assert agents_pos.shape == (2, 2)
    assert agents_fit.shape == (2,)
    assert best_pos.shape == (2, 2)
    assert best_fit.shape == (2,)
    assert history.get_convergence("local_position").shape == (2,)
    assert np.array_equal(history.get_convergence("value"), [1, 2])


def test_history_skips_agents_when_disabled():
    history = History()

    history.dump(agents=make_agents())

    assert not hasattr(history, "agents")


def test_history_annotations_describe_both_return_shapes_and_integer_indexes():
    hints = get_type_hints(History.get_convergence)

    assert hints["return"] == (np.ndarray | tuple[np.ndarray, np.ndarray])
    assert hints["index"] == (int | tuple[int, ...] | None)


@pytest.mark.parametrize("n_dimensions", [1, 2])
def test_history_preserves_snapshots_and_concatenation_shapes(n_dimensions):
    agents = [Agent(2, n_dimensions, [0, 0], [1000, 1000]) for _ in range(2)]
    first = np.arange(1, 2 * n_dimensions + 1).reshape(2, n_dimensions)
    history = History(save_agents=True)
    for iteration in range(2):
        for index, agent in enumerate(agents):
            agent.position[:] = first + index * 100 + iteration * 10
            agent.fit = float(agent.position.sum())
        history.dump(
            agents=agents,
            best_agent=agents[0],
            local_position=np.array([agent.position for agent in agents]),
        )

    positions, fitness = history.get_convergence("agents", index=1)
    expected = np.hstack((first + 100, first + 110))
    np.testing.assert_array_equal(positions, expected)
    np.testing.assert_array_equal(fitness, [(first + 100).sum(), (first + 110).sum()])
    np.testing.assert_array_equal(history.get_convergence("local_position", index=1), expected)
    np.testing.assert_array_equal(history.best_agent[0][0], first)


def test_history_missing_keys_are_not_silently_defaulted():
    with pytest.raises(AttributeError):
        History().get_convergence("best_agent")
