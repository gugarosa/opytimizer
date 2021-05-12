import numpy as np

from opytimizer.optimizers.swarm import sca
from opytimizer.spaces import search


def test_sca_params():
    params = {
        'r_min': 0,
        'r_max': 2,
        'a': 3,
    }

    new_sca = sca.SCA(params=params)

    assert new_sca.r_min == 0

    assert new_sca.r_max == 2

    assert new_sca.a == 3


def test_sca_params_setter():
    new_sca = sca.SCA()

    try:
        new_sca.r_min = 'a'
    except:
        new_sca.r_min = 0.1

    try:
        new_sca.r_min = -1
    except:
        new_sca.r_min = 0.1

    assert new_sca.r_min == 0.1

    try:
        new_sca.r_max = 'b'
    except:
        new_sca.r_max = 2

    try:
        new_sca.r_max = -1
    except:
        new_sca.r_max = 2

    try:
        new_sca.r_max = 0
    except:
        new_sca.r_max = 2

    assert new_sca.r_max == 2

    try:
        new_sca.a = 'c'
    except:
        new_sca.a = 0.5

    try:
        new_sca.a = -1
    except:
        new_sca.a = 0.5

    assert new_sca.a == 0.5


def test_sca_update_position():
    new_sca = sca.SCA()

    position = new_sca._update_position(1, 1, 0.5, 0.5, 0.5, 0.5)

    assert position > 0


def test_sca_update():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_sca = sca.SCA()

    new_sca.update(search_space, 1, 10)
