import numpy as np

from opytimizer.optimizers.swarm import fpa
from opytimizer.spaces import search


def test_fpa_params():
    params = {"beta": 1.0, "eta": 0.5, "p": 0.5}

    new_fpa = fpa.FPA(params=params)

    assert new_fpa.beta == 1.0

    assert new_fpa.eta == 0.5

    assert new_fpa.p == 0.5


def test_fpa_params_setter():
    new_fpa = fpa.FPA()

    try:
        new_fpa.beta = "a"
    except:
        new_fpa.beta = 0.75

    try:
        new_fpa.beta = -1
    except:
        new_fpa.beta = 0.75

    assert new_fpa.beta == 0.75

    try:
        new_fpa.eta = "b"
    except:
        new_fpa.eta = 1.5

    try:
        new_fpa.eta = -1
    except:
        new_fpa.eta = 1.5

    assert new_fpa.eta == 1.5

    try:
        new_fpa.p = "c"
    except:
        new_fpa.p = 0.25

    try:
        new_fpa.p = -1
    except:
        new_fpa.p = 0.25

    assert new_fpa.p == 0.25


def test_fpa_global_pollination():
    new_fpa = fpa.FPA()

    position = new_fpa._global_pollination(1, 2)

    assert position != 0


def test_fpa_local_pollination():
    new_fpa = fpa.FPA()

    position = new_fpa._local_pollination(1, 2, 1, 0.5)

    assert position == 1.5


def test_fpa_update():
    def square(x):
        return np.sum(x**2)

    new_fpa = fpa.FPA()

    search_space = search.SearchSpace(
        n_agents=2, n_variables=2, lower_bound=[1, 1], upper_bound=[10, 10]
    )

    new_fpa.update(search_space, square)

    new_fpa.p = 0.01
    new_fpa.update(search_space, square)
