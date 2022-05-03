import numpy as np

from opytimizer.optimizers.swarm import stoa
from opytimizer.spaces import search


def test_stoa_params():
    params = {"Cf": 2.0, "u": 1.0, "v": 1.0}

    new_stoa = stoa.STOA(params=params)

    assert new_stoa.Cf == 2.0

    assert new_stoa.u == 1.0

    assert new_stoa.v == 1.0


def test_stoa_params_setter():
    new_stoa = stoa.STOA()

    try:
        new_stoa.Cf = "a"
    except:
        new_stoa.Cf = 2.0

    assert new_stoa.Cf == 2.0

    try:
        new_stoa.Cf = -1
    except:
        new_stoa.Cf = 2.0

    assert new_stoa.Cf == 2.0

    try:
        new_stoa.u = "b"
    except:
        new_stoa.u = 1.0

    assert new_stoa.u == 1.0

    try:
        new_stoa.u = -1
    except:
        new_stoa.u = 1.0

    assert new_stoa.u == 1.0

    try:
        new_stoa.v = "b"
    except:
        new_stoa.v = 1.0

    assert new_stoa.v == 1.0

    try:
        new_stoa.v = -1
    except:
        new_stoa.v = 1.0

    assert new_stoa.v == 1.0


def test_stoa_update():
    search_space = search.SearchSpace(
        n_agents=2, n_variables=2, lower_bound=[1, 1], upper_bound=[10, 10]
    )

    new_stoa = stoa.STOA()

    new_stoa.update(search_space, 1, 10)
