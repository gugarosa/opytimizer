import numpy as np

from opytimizer.optimizers.swarm import ba
from opytimizer.spaces import search


def test_ba_params():
    params = {
        'f_min': 0,
        'f_max': 2,
        'A': 0.5,
        'r': 0.5
    }

    new_ba = ba.BA(params=params)

    assert new_ba.f_min == 0

    assert new_ba.f_max == 2

    assert new_ba.A == 0.5

    assert new_ba.r == 0.5


def test_ba_params_setter():
    new_ba = ba.BA()

    try:
        new_ba.f_min = 'a'
    except:
        new_ba.f_min = 0.1

    try:
        new_ba.f_min = -1
    except:
        new_ba.f_min = 0.1

    assert new_ba.f_min == 0.1

    try:
        new_ba.f_max = 'b'
    except:
        new_ba.f_max = 2

    try:
        new_ba.f_max = -1
    except:
        new_ba.f_max = 2

    try:
        new_ba.f_max = 0
    except:
        new_ba.f_max = 2

    assert new_ba.f_max == 2

    try:
        new_ba.A = 'c'
    except:
        new_ba.A = 0.5

    try:
        new_ba.A = -1
    except:
        new_ba.A = 0.5

    assert new_ba.A == 0.5

    try:
        new_ba.r = 'd'
    except:
        new_ba.r = 0.5

    try:
        new_ba.r = -1
    except:
        new_ba.r = 0.5

    assert new_ba.r == 0.5


def test_ba_compile():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_ba = ba.BA()
    new_ba.compile(search_space)

    try:
        new_ba.frequency = 1
    except:
        new_ba.frequency = np.array([1])

    assert new_ba.frequency == np.array([1])

    try:
        new_ba.velocity = 1
    except:
        new_ba.velocity = np.array([1])

    assert new_ba.velocity == np.array([1])

    try:
        new_ba.loudness = 1
    except:
        new_ba.loudness = np.array([1])

    assert new_ba.loudness == np.array([1])

    try:
        new_ba.pulse_rate = 1
    except:
        new_ba.pulse_rate = np.array([1])

    assert new_ba.pulse_rate == np.array([1])


def test_ba_update():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_ba = ba.BA()
    new_ba.compile(search_space)

    new_ba.update(search_space, square, 1)
