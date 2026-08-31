import numpy as np
import pytest

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
