import numpy as np
import pytest

from opytimizer.spaces import pareto


def test_pareto_space_loads_agents_synchronously():
    data_points = np.arange(6).reshape(2, 3)
    space = pareto.ParetoSpace(data_points)

    assert len(space.agents) == 2
    assert np.array_equal(space.agents[1].position[:, 0], data_points[1])
    assert not hasattr(space, "built")


def test_pareto_space_validates_data_points():
    with pytest.raises(TypeError):
        pareto.ParetoSpace([[1, 2]])
    with pytest.raises(ValueError):
        pareto.ParetoSpace(np.array([]))
