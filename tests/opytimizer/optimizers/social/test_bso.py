import numpy as np

from opytimizer.optimizers.social import bso
from opytimizer.spaces import search


def test_bso_params():
    params = {
        'm': 5,
        'p_replacement_cluster': 0.2,
        'p_single_cluster': 0.8,
        'p_single_best': 0.4,
        'p_double_best': 0.5,
        'k': 20
    }

    new_bso = bso.BSO(params=params)

    assert new_bso.m == 5

    assert new_bso.p_replacement_cluster == 0.2

    assert new_bso.p_single_cluster == 0.8

    assert new_bso.p_single_best == 0.4

    assert new_bso.p_double_best == 0.5

    assert new_bso.k == 20


def test_bso_params_setter():
    new_bso = bso.BSO()

    try:
        new_bso.m = 'a'
    except:
        new_bso.m = 5

    assert new_bso.m == 5

    try:
        new_bso.m = -1
    except:
        new_bso.m = 5

    assert new_bso.m == 5

    try:
        new_bso.p_replacement_cluster = 'b'
    except:
        new_bso.p_replacement_cluster = 0.2

    assert new_bso.p_replacement_cluster == 0.2

    try:
        new_bso.p_replacement_cluster = -1
    except:
        new_bso.p_replacement_cluster = 0.2

    assert new_bso.p_replacement_cluster == 0.2

    try:
        new_bso.p_single_cluster = 'c'
    except:
        new_bso.p_single_cluster = 0.8

    assert new_bso.p_single_cluster == 0.8

    try:
        new_bso.p_single_cluster = -1
    except:
        new_bso.p_single_cluster = 0.8

    assert new_bso.p_single_cluster == 0.8

    try:
        new_bso.p_single_best = 'd'
    except:
        new_bso.p_single_best = 0.4

    assert new_bso.p_single_best == 0.4

    try:
        new_bso.p_single_best = -1
    except:
        new_bso.p_single_best = 0.4

    assert new_bso.p_single_best == 0.4

    try:
        new_bso.p_double_best = 'e'
    except:
        new_bso.p_double_best = 0.5

    assert new_bso.p_double_best == 0.5

    try:
        new_bso.p_double_best = -1
    except:
        new_bso.p_double_best = 0.5

    assert new_bso.p_double_best == 0.5

    try:
        new_bso.k = 'f'
    except:
        new_bso.k = 20

    assert new_bso.k == 20

    try:
        new_bso.k = -1
    except:
        new_bso.k = 20

    assert new_bso.k == 20


def test_bso_clusterize():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_bso = bso.BSO()

    new_bso._clusterize(search_space.agents)


def test_bso_sigmoid():
    new_bso = bso.BSO()

    x = 0.5

    y = new_bso._sigmoid(x)

    assert y == 0.6224593312018546


def test_bso_update():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=50, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_bso = bso.BSO()
    new_bso.evaluate(search_space, square)

    new_bso.update(search_space, square, 1, 10)

    new_bso.p_replacement_cluster = 1
    new_bso.update(search_space, square, 1, 10)
