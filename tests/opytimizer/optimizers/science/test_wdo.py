import numpy as np

from opytimizer.optimizers.science import wdo
from opytimizer.spaces import search


def test_wdo_params():
    params = {"v_max": 0.3, "alpha": 0.8, "g": 0.6, "c": 1.0, "RT": 1.5}

    new_wdo = wdo.WDO(params=params)

    assert new_wdo.v_max == 0.3

    assert new_wdo.alpha == 0.8

    assert new_wdo.g == 0.6

    assert new_wdo.c == 1.0

    assert new_wdo.RT == 1.5


def test_wdo_params_setter():
    new_wdo = wdo.WDO()

    try:
        new_wdo.v_max = "a"
    except:
        new_wdo.v_max = 0.1

    try:
        new_wdo.v_max = -1
    except:
        new_wdo.v_max = 0.1

    assert new_wdo.v_max == 0.1

    try:
        new_wdo.alpha = "b"
    except:
        new_wdo.alpha = 0.8

    try:
        new_wdo.alpha = -1
    except:
        new_wdo.alpha = 0.8

    assert new_wdo.alpha == 0.8

    try:
        new_wdo.g = "c"
    except:
        new_wdo.g = 0.5

    try:
        new_wdo.g = -1
    except:
        new_wdo.g = 0.5

    assert new_wdo.g == 0.5

    try:
        new_wdo.c = "d"
    except:
        new_wdo.c = 0.5

    try:
        new_wdo.c = -1
    except:
        new_wdo.c = 0.5

    assert new_wdo.c == 0.5

    try:
        new_wdo.RT = "e"
    except:
        new_wdo.RT = 0.5

    try:
        new_wdo.RT = -1
    except:
        new_wdo.RT = 0.5

    assert new_wdo.RT == 0.5


def test_wdo_compile():
    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_wdo = wdo.WDO()
    new_wdo.compile(search_space)

    try:
        new_wdo.velocity = 1
    except:
        new_wdo.velocity = np.array([1])

    assert new_wdo.velocity == np.array([1])


def test_wdo_update():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_wdo = wdo.WDO()
    new_wdo.compile(search_space)

    new_wdo.update(search_space, square)
