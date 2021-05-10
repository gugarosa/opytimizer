import numpy as np

from opytimizer.optimizers.population import ao
from opytimizer.spaces import search


def test_ao_params():
    params = {
        'alpha': 0.1,
        'delta': 0.1,
        'n_cycles': 10,
        'U': 0.00565,
        'w': 0.005
    }

    new_ao = ao.AO(params=params)

    assert new_ao.alpha == 0.1

    assert new_ao.delta == 0.1

    assert new_ao.n_cycles == 10

    assert new_ao.U == 0.00565

    assert new_ao.w == 0.005


def test_ao_params_setter():
    new_ao = ao.AO()

    try:
        new_ao.alpha = 'a'
    except:
        new_ao.alpha = 0.1

    try:
        new_ao.alpha = -1
    except:
        new_ao.alpha = 0.1

    assert new_ao.alpha == 0.1

    try:
        new_ao.delta = 'b'
    except:
        new_ao.delta = 0.1

    try:
        new_ao.delta = -1
    except:
        new_ao.delta = 0.1

    try:
        new_ao.n_cycles = 'c'
    except:
        new_ao.n_cycles = 10

    try:
        new_ao.n_cycles = -1
    except:
        new_ao.n_cycles = 10

    assert new_ao.n_cycles == 10

    try:
        new_ao.U = 'd'
    except:
        new_ao.U = 0.00565

    try:
        new_ao.U = -1
    except:
        new_ao.U = 0.00565

    assert new_ao.U == 0.00565

    try:
        new_ao.w = 'e'
    except:
        new_ao.w = 0.005

    try:
        new_ao.w = -1
    except:
        new_ao.w = 0.005

    assert new_ao.w == 0.005


def test_ao_update():
    def square(x):
        return np.sum(x**2)

    new_ao = ao.AO()

    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_ao.update(search_space, square, 1, 10)
    new_ao.update(search_space, square, 8, 10)
