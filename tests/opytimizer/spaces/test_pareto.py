import numpy as np

from opytimizer.spaces import pareto


def test_pareto_space_load_agents():
    data_points = np.zeros((10, 3))

    new_pareto_space = pareto.ParetoSpace(data_points)

    new_pareto_space._load_agents(data_points)

    assert len(new_pareto_space.agents) == 10


def test_pareto_space_build():
    data_points = np.zeros((10, 3))

    new_pareto_space = pareto.ParetoSpace(data_points)

    new_pareto_space.build(data_points)

    assert new_pareto_space.built == True
