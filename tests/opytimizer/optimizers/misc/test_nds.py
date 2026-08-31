import numpy as np

import opytimizer.math.random as r
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
    data_points = r.generate_uniform_random_number(size=(10, 3))

    search_space = pareto.ParetoSpace(data_points)

    new_nds = nds.NDS()
    new_nds.compile(search_space)

    new_nds.update(search_space)
