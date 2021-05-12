import numpy as np

from opytimizer.optimizers.swarm import sbo
from opytimizer.spaces import search


def test_sbo_params():
    params = {
        'alpha': 0.9,
        'p_mutation': 0.05,
        'z': 0.02
    }

    new_sbo = sbo.SBO(params=params)

    assert new_sbo.alpha == 0.9

    assert new_sbo.p_mutation == 0.05

    assert new_sbo.z == 0.02


def test_sbo_params_setter():
    new_sbo = sbo.SBO()

    try:
        new_sbo.alpha = 'a'
    except:
        new_sbo.alpha = 0.75

    try:
        new_sbo.alpha = -1
    except:
        new_sbo.alpha = 0.75

    assert new_sbo.alpha == 0.75

    try:
        new_sbo.p_mutation = 'b'
    except:
        new_sbo.p_mutation = 0.05

    try:
        new_sbo.p_mutation = 1.5
    except:
        new_sbo.p_mutation = 0.05

    assert new_sbo.p_mutation == 0.05

    try:
        new_sbo.z = 'c'
    except:
        new_sbo.z = 0.02

    try:
        new_sbo.z = 1.5
    except:
        new_sbo.z = 0.02

    assert new_sbo.z == 0.02


def test_sbo_create_additional_attrs():
    search_space = search.SearchSpace(n_agents=2, n_variables=2,
                                      lower_bound=[1, 1], upper_bound=[10, 10])

    new_sbo = sbo.SBO()
    new_sbo.create_additional_attrs(search_space)

    try:
        new_sbo.sigma = 1
    except:
        new_sbo.sigma = []

    assert new_sbo.sigma == []


def test_sbo_update():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=2, n_variables=2,
                                      lower_bound=[1, 1], upper_bound=[10, 10])

    new_sbo = sbo.SBO()
    new_sbo.create_additional_attrs(search_space)
    new_sbo.p_mutation = 1

    new_sbo.update(search_space, square)
