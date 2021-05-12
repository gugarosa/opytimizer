import numpy as np

from opytimizer.optimizers.swarm import abo
from opytimizer.spaces import search


def test_abo_params():
    params = {
        'sunspot_ratio': 0.9,
        'a': 2.0
    }

    new_abo = abo.ABO(params=params)

    assert new_abo.sunspot_ratio == 0.9

    assert new_abo.a == 2.0


def test_abo_params_setter():
    new_abo = abo.ABO()

    try:
        new_abo.sunspot_ratio = 'a'
    except:
        new_abo.sunspot_ratio = 0.9

    try:
        new_abo.sunspot_ratio = -1
    except:
        new_abo.sunspot_ratio = 0.9

    assert new_abo.sunspot_ratio == 0.9

    try:
        new_abo.a = 'b'
    except:
        new_abo.a = 2.0

    try:
        new_abo.a = -1
    except:
        new_abo.a = 2.0

    assert new_abo.a == 2.0


def test_abo_flight_mode():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_abo = abo.ABO()

    new_abo._flight_mode(
        search_space.agents[0], search_space.agents[1], square)


def test_abo_update():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_abo = abo.ABO()

    new_abo.update(search_space, square, 1, 10)
    new_abo.update(search_space, square, 5, 10)
