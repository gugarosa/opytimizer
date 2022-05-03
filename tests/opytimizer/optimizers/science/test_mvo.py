import numpy as np

from opytimizer.optimizers.science import mvo
from opytimizer.spaces import search


def test_mvo_params():
    params = {"WEP_min": 0.2, "WEP_max": 1.0, "p": 0.5}

    new_mvo = mvo.MVO(params=params)

    assert new_mvo.WEP_min == 0.2

    assert new_mvo.WEP_max == 1.0

    assert new_mvo.p == 0.5


def test_mvo_params_setter():
    new_mvo = mvo.MVO()

    try:
        new_mvo.WEP_min = "a"
    except:
        new_mvo.WEP_min = 0.75

    try:
        new_mvo.WEP_min = -1
    except:
        new_mvo.WEP_min = 0.75

    assert new_mvo.WEP_min == 0.75

    try:
        new_mvo.WEP_max = "b"
    except:
        new_mvo.WEP_max = 0.9

    try:
        new_mvo.WEP_max = 0.1
    except:
        new_mvo.WEP_max = 0.9

    try:
        new_mvo.WEP_max = -1
    except:
        new_mvo.WEP_max = 0.9

    assert new_mvo.WEP_max == 0.9

    try:
        new_mvo.p = "c"
    except:
        new_mvo.p = 0.25

    try:
        new_mvo.p = -1
    except:
        new_mvo.p = 0.25

    assert new_mvo.p == 0.25


def test_mvo_update():
    def square(x):
        return np.sum(x**2)

    new_mvo = mvo.MVO()

    search_space = search.SearchSpace(
        n_agents=2, n_variables=2, lower_bound=[1, 1], upper_bound=[10, 10]
    )

    new_mvo.update(search_space, square, 1, 10)
    new_mvo.update(search_space, square, 5, 10)
