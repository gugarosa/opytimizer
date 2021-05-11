import numpy as np

from opytimizer.optimizers.science import eo
from opytimizer.spaces import search


def test_eo_params():
    params = {
        'a1': 2.0,
        'a2': 1.0,
        'GP': 0.5,
        'V': 1.0
    }

    new_eo = eo.EO(params=params)

    assert new_eo.a1 == 2.0

    assert new_eo.a2 == 1.0

    assert new_eo.GP == 0.5

    assert new_eo.V == 1.0


def test_eo_params_setter():
    new_eo = eo.EO()

    try:
        new_eo.a1 = 'a'
    except:
        new_eo.a1 = 2.0

    try:
        new_eo.a1 = -1
    except:
        new_eo.a1 = 2.0

    assert new_eo.a1 == 2.0

    try:
        new_eo.a2 = 'b'
    except:
        new_eo.a2 = 1.0

    try:
        new_eo.a2 = -1
    except:
        new_eo.a2 = 1.0

    assert new_eo.a2 == 1.0

    try:
        new_eo.GP = 'c'
    except:
        new_eo.GP = 0.5

    try:
        new_eo.GP = -1
    except:
        new_eo.GP = 0.5

    assert new_eo.GP == 0.5

    try:
        new_eo.V = 'd'
    except:
        new_eo.V = 1.0

    try:
        new_eo.V = -1
    except:
        new_eo.V = 1.0

    assert new_eo.V == 1.0


def test_eo_create_additional_attrs():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[1, 1], upper_bound=[10, 10])

    new_eo = eo.EO()
    new_eo.create_additional_attrs(search_space)

    try:
        new_eo.C = 1
    except:
        new_eo.C = []

    assert new_eo.C == []


def test_eo_calculate_equilibrium():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[1, 1], upper_bound=[10, 10])

    new_eo = eo.EO()
    new_eo.create_additional_attrs(search_space)

    new_eo._calculate_equilibrium(search_space.agents)


def test_eo_average_concentration():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[1, 1], upper_bound=[10, 10])

    new_eo = eo.EO()
    new_eo.create_additional_attrs(search_space)

    C_avg = new_eo._average_concentration(square)

    assert type(C_avg).__name__ == 'Agent'


def test_eo_update():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[1, 1], upper_bound=[10, 10])

    new_eo = eo.EO()
    new_eo.create_additional_attrs(search_space)

    new_eo.update(search_space, square, 1, 10)
