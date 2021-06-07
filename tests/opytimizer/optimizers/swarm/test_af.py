import numpy as np

from opytimizer.optimizers.swarm import af
from opytimizer.spaces import search


def test_af_params():
    params = {
        'c1': 0.75,
        'c2': 1.25,
        'm': 10,
        'Q': 0.75
    }

    new_af = af.AF(params=params)

    assert new_af.c1 == 0.75

    assert new_af.c2 == 1.25

    assert new_af.m == 10

    assert new_af.Q == 0.75


def test_af_params_setter():
    new_af = af.AF()

    try:
        new_af.c1 = 'a'
    except:
        new_af.c1 = 0.75

    assert new_af.c1 == 0.75

    try:
        new_af.c1 = -1
    except:
        new_af.c1 = 0.75

    assert new_af.c1 == 0.75

    try:
        new_af.c2 = 'b'
    except:
        new_af.c2 = 1.25

    assert new_af.c2 == 1.25

    try:
        new_af.c2 = -1
    except:
        new_af.c2 = 1.25

    assert new_af.c2 == 1.25

    try:
        new_af.m = 'c'
    except:
        new_af.m = 10

    assert new_af.m == 10

    try:
        new_af.m = 0
    except:
        new_af.m = 10

    assert new_af.m == 10

    try:
        new_af.Q = 'd'
    except:
        new_af.Q = 0.75

    assert new_af.Q == 0.75

    try:
        new_af.Q = -1
    except:
        new_af.Q = 0.75

    assert new_af.Q == 0.75


def test_af_compile():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_af = af.AF()
    new_af.compile(search_space)

    try:
        new_af.p_distance = 1
    except:
        new_af.p_distance = np.array([1])

    assert new_af.p_distance == np.array([1])

    try:
        new_af.g_distance = 1
    except:
        new_af.g_distance = np.array([1])

    assert new_af.g_distance == np.array([1])


def test_af_update():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_af = af.AF()
    new_af.compile(search_space)

    new_af.evaluate(search_space, square)
    new_af.update(search_space, square)
