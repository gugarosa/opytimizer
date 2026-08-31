import numpy as np

from opytimizer.optimizers.misc import nds
from opytimizer.spaces import pareto


def test_nds_params():
    new_nds = nds.NDS()

    assert new_nds.n_pareto_points == 0


def test_nds_compile():
    data_points = np.zeros((10, 3))

    search_space = pareto.ParetoSpace(data_points)

    new_nds = nds.NDS()
    new_nds.compile(search_space)


def test_nds_update_1():
    data_points = np.zeros((10, 3))

    search_space = pareto.ParetoSpace(data_points)

    new_nds = nds.NDS()
    new_nds.compile(search_space)

    new_nds.update(search_space)


def test_nds_update_2():
    data_points = np.random.uniform(size=(10, 3))

    search_space = pareto.ParetoSpace(data_points)

    new_nds = nds.NDS()
    new_nds.compile(search_space)

    new_nds.update(search_space)
