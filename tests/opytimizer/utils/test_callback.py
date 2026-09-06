# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

from pathlib import Path
from types import SimpleNamespace
from typing import get_type_hints

import numpy as np
import pytest

from opytimizer import Opytimizer
from opytimizer.core import Optimizer
from opytimizer.spaces import HyperComplexSpace, SearchSpace
from opytimizer.utils.callback import (
    Callback,
    CheckpointCallback,
    DiscreteSearchCallback,
)


def test_callback_hooks_are_noops():
    callback = Callback()

    assert callback.on_task_begin(None) is None
    assert callback.on_task_end(None) is None
    assert callback.on_iteration_begin(1, None) is None
    assert callback.on_iteration_end(1, None) is None
    assert callback.on_evaluate_before() is None
    assert callback.on_evaluate_after() is None
    assert callback.on_update_before() is None
    assert callback.on_update_after() is None


def test_callback_annotations_refer_to_the_actual_model():
    assert get_type_hints(Callback.on_task_begin)["opt_model"] is Opytimizer
    assert get_type_hints(CheckpointCallback.on_iteration_end)["opt_model"] is Opytimizer
    assert get_type_hints(DiscreteSearchCallback.on_task_begin)["opt_model"] is Opytimizer


def test_checkpoint_callback_saves_on_frequency():
    saved = []
    model = SimpleNamespace(save=saved.append)
    callback = CheckpointCallback("model.pkl", frequency=2)

    callback.on_iteration_end(1, model)
    callback.on_iteration_end(2, model)

    assert saved == ["iter_2_model.pkl"]


@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize("path_type", [str, Path])
def test_checkpoint_callback_preserves_directory_and_saves_state(tmp_path, monkeypatch, relative, path_type):
    (tmp_path / "checkpoints").mkdir()
    monkeypatch.chdir(tmp_path)
    path = Path("checkpoints") / "model.pkl"
    if not relative:
        path = tmp_path / path
    model = Opytimizer(SearchSpace(1, 1, 0, 1), Optimizer(), lambda x: float(np.sum(x**2)))

    callback = CheckpointCallback(path_type(path), frequency=1)
    assert callback.file_path == str(path)
    model.start(1, [callback])

    checkpoint = path.with_name("iter_1_model.pkl")
    assert checkpoint.is_file()
    loaded = Opytimizer.load(checkpoint)
    assert loaded.total_iterations == 1
    assert loaded.history.best_agent == model.history.best_agent
    np.testing.assert_array_equal(loaded.space.best_agent.position, model.space.best_agent.position)
    assert loaded.function(np.array([2])) == 4


@pytest.mark.parametrize(
    "args,error",
    [
        ((1,), TypeError),
        (("model.pkl", 1.0), TypeError),
        (("model.pkl", -1), ValueError),
    ],
)
def test_checkpoint_callback_validates_constructor_inputs(args, error):
    with pytest.raises(error):
        CheckpointCallback(*args)


def test_discrete_search_callback_validates_space_values():
    space = SearchSpace(1, 2, [0, 0], [1, 1])
    model = SimpleNamespace(space=space)

    DiscreteSearchCallback([[0, 1], [0, 1]]).on_task_begin(model)

    with pytest.raises(ValueError):
        DiscreteSearchCallback([[0, 1]]).on_task_begin(model)
    with pytest.raises(ValueError):
        DiscreteSearchCallback([[0, 2], [0, 1]]).on_task_begin(model)


@pytest.mark.parametrize("values", [[], 0, [[0, 1]]])
def test_discrete_search_callback_rejects_empty_or_non_vector_values(values):
    model = SimpleNamespace(space=SearchSpace(1, 1, 0, 1))

    with pytest.raises(ValueError, match="non-empty"):
        DiscreteSearchCallback([values]).on_task_begin(model)


@pytest.mark.parametrize(
    "position,expected",
    [
        ([[0.2, 0.8]], [[0, 1]]),
        ([[0.2, 0.8, 0.5]], [[0, 1, 0]]),
    ],
)
def test_discrete_search_callback_projects_each_dimension(position, expected):
    space = HyperComplexSpace(1, 1, len(position[0]))
    space.agents[0].position[:] = position
    model = Opytimizer(space, Optimizer(), lambda x: float(np.sum(x**2)))

    model.start(0, [DiscreteSearchCallback([[0, 1]])])

    np.testing.assert_array_equal(space.agents[0].position, expected)
    np.testing.assert_array_equal(space.best_agent.position, expected)
    assert space.best_agent.fit == np.sum(np.asarray(expected) ** 2)


def test_discrete_search_callback_maps_to_nearest_values():
    space = SearchSpace(1, 2, [0, 0], [1, 1])
    space.agents[0].position[:, 0] = [0.2, 0.8]
    callback = DiscreteSearchCallback([[0, 1], [0, 1]])

    callback.on_evaluate_before(space)

    assert np.array_equal(space.agents[0].position[:, 0], [0, 1])

    with pytest.raises(TypeError):
        callback.on_evaluate_before(None)


def test_discrete_search_callback_requires_list():
    with pytest.raises(TypeError):
        DiscreteSearchCallback(1)
