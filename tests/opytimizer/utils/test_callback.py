from types import SimpleNamespace

import numpy as np
import pytest

from opytimizer.spaces import SearchSpace
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


def test_checkpoint_callback_saves_on_frequency():
    saved = []
    model = SimpleNamespace(save=saved.append)
    callback = CheckpointCallback("model.pkl", frequency=2)

    callback.on_iteration_end(1, model)
    callback.on_iteration_end(2, model)

    assert saved == ["iter_2_model.pkl"]


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
