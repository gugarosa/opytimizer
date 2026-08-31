import numpy as np
import pytest

from opytimizer.core import Agent, Space


def test_space_initializes_configuration_without_lifecycle_flag():
    space = Space(2, 2, 3, [-1, -2], [1, 2], ["a", "b"])

    assert space.agents == []
    assert isinstance(space.best_agent, Agent)
    assert space.mapping == ["a", "b"]
    assert not hasattr(space, "built")


@pytest.mark.parametrize(
    "args,error",
    [
        ((1.0,), TypeError),
        ((0,), ValueError),
        ((1, 0), ValueError),
        ((1, 1, 0), ValueError),
        ((1, 2, 1, [0], [1, 1]), ValueError),
        ((1, 2, 1, [0, 0], [1]), ValueError),
        ((1, 1, 1, 0, 1, "x"), TypeError),
    ],
)
def test_space_validates_constructor_inputs(args, error):
    with pytest.raises(error):
        Space(*args)


def test_space_builds_and_clips_agents():
    space = Space(2, 1, 1, 0, 1)

    space.build()
    space.agents[0].position[:] = -1
    space.agents[1].position[:] = 2
    space.clip_by_bound()

    assert len(space.agents) == 2
    assert np.array_equal(space.agents[0].position, [[0]])
    assert np.array_equal(space.agents[1].position, [[1]])
