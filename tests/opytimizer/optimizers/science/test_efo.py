import numpy as np

from opytimizer.optimizers.science import efo
from opytimizer.spaces import search

np.random.seed(0)


def test_efo_params():
    params = {
        "positive_field": 0.1,
        "negative_field": 0.5,
        "ps_ratio": 0.1,
        "r_ratio": 0.4,
    }

    new_efo = efo.EFO(params=params)

    assert new_efo.positive_field == 0.1

    assert new_efo.negative_field == 0.5

    assert new_efo.ps_ratio == 0.1

    assert new_efo.r_ratio == 0.4

    assert new_efo.phi == (1 + np.sqrt(5)) / 2

    assert new_efo.RI == 0


def test_efo_params_setter():
    new_efo = efo.EFO()

    try:
        new_efo.positive_field = "a"
    except:
        new_efo.positive_field = 0.5

    try:
        new_efo.positive_field = -1
    except:
        new_efo.positive_field = 0.5

    assert new_efo.positive_field == 0.5

    try:
        new_efo.negative_field = "b"
    except:
        new_efo.negative_field = 0.2

    try:
        new_efo.negative_field = 0.99
    except:
        new_efo.negative_field = 0.2

    try:
        new_efo.negative_field = -1
    except:
        new_efo.negative_field = 0.2

    assert new_efo.negative_field == 0.2

    try:
        new_efo.ps_ratio = "c"
    except:
        new_efo.ps_ratio = 0.25

    try:
        new_efo.ps_ratio = -1
    except:
        new_efo.ps_ratio = 0.25

    assert new_efo.ps_ratio == 0.25

    try:
        new_efo.r_ratio = "d"
    except:
        new_efo.r_ratio = 0.25

    try:
        new_efo.r_ratio = -1
    except:
        new_efo.r_ratio = 0.25

    assert new_efo.r_ratio == 0.25

    try:
        new_efo.phi = "e"
    except:
        new_efo.phi = (1 + np.sqrt(5)) / 2

    assert new_efo.phi == (1 + np.sqrt(5)) / 2

    try:
        new_efo.RI = "f"
    except:
        new_efo.RI = 0

    try:
        new_efo.RI = -1
    except:
        new_efo.RI = 0

    assert new_efo.RI == 0


def test_efo_calculate_indexes():
    new_efo = efo.EFO()

    a, b, c = new_efo._calculate_indexes(30)

    assert a >= 0
    assert b >= 0
    assert c >= 0


def test_efo_update():
    def square(x):
        return np.sum(x**2)

    new_efo = efo.EFO()

    search_space = search.SearchSpace(
        n_agents=2, n_variables=2, lower_bound=[1, 1], upper_bound=[10, 10]
    )

    new_efo.update(search_space, square)
