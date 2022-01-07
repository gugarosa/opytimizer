import numpy as np

from opytimizer.optimizers.misc import nds
from opytimizer.spaces import pareto

import opytimizer.math.random as r


def test_nds_params():
    new_nds = nds.NDS()

    assert new_nds.n_pareto_points == 0


def test_nds_params_setter():
    new_nds = nds.NDS()

    try:
        new_nds.n_pareto_points = 'a'
    except:
        new_nds.n_pareto_points = 0

    assert new_nds.n_pareto_points == 0

    try:
        new_nds.n_pareto_points = -1
    except:
        new_nds.n_pareto_points = 0

    assert new_nds.n_pareto_points == 0


def test_nds_compile():
    data_points = np.zeros((10, 3))

    search_space = pareto.ParetoSpace(data_points)

    new_nds = nds.NDS()
    new_nds.compile(search_space)

    try:
        new_nds.count = 1
    except:
        new_nds.count = np.array([1])

    assert new_nds.count == 1

    try:
        new_nds.set = 1
    except:
        new_nds.set = np.array([1])

    assert new_nds.set == 1

    try:
        new_nds.status = 1
    except:
        new_nds.status = np.array([1])

    assert new_nds.status == 1


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
