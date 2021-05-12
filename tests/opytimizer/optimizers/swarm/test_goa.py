import numpy as np

from opytimizer.optimizers.swarm import goa
from opytimizer.spaces import search

np.random.seed(0)


def test_goa_params():
    params = {
        'c_min': 0.00001,
        'c_max': 1.0,
        'f': 0.5,
        'l': 1.5
    }

    new_goa = goa.GOA(params=params)

    assert new_goa.c_min == 0.00001

    assert new_goa.c_max == 1.0

    assert new_goa.f == 0.5

    assert new_goa.l == 1.5


def test_goa_params_setter():
    new_goa = goa.GOA()

    try:
        new_goa.c_min = 'a'
    except:
        new_goa.c_min = 0.00001

    try:
        new_goa.c_min = -1
    except:
        new_goa.c_min = 0.00001

    assert new_goa.c_min == 0.00001

    try:
        new_goa.c_max = 'b'
    except:
        new_goa.c_max = 2.0

    try:
        new_goa.c_max = 0
    except:
        new_goa.c_max = 1.0

    assert new_goa.c_max == 1.0

    try:
        new_goa.f = 'c'
    except:
        new_goa.f = 0.5

    try:
        new_goa.f = -1
    except:
        new_goa.f = 0.5

    assert new_goa.f == 0.5

    try:
        new_goa.l = 'd'
    except:
        new_goa.l = 1.5

    try:
        new_goa.l = -1
    except:
        new_goa.l = 1.5

    assert new_goa.l == 1.5


def test_goa_social_force():
    new_goa = goa.GOA()

    r = new_goa._social_force(np.array([1, 1, 1]))

    assert r[0] == -0.11117088165514633


def test_goa_update():
    def square(x):
        return np.sum(x**2)

    new_goa = goa.GOA()

    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_goa.update(search_space, square, 1, 10)
