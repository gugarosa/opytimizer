import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.swarm import ba
from opytimizer.spaces import search
from opytimizer.utils import constant

np.random.seed(0)


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


def test_ba_build():
    new_ba = ba.BA()

    assert new_ba.built == True


def test_ba_update_frequency():
    new_ba = ba.BA()

    frequency = new_ba._update_frequency(0, 2)

    assert frequency != 0


def test_ba_update_velocity():
    new_ba = ba.BA()

    velocity = new_ba._update_velocity(1, 1, 1, 1)

    assert velocity != 0


def test_ba_update_position():
    new_ba = ba.BA()

    position = new_ba._update_position(1, 1)

    assert position == 2


def test_ba_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    params = {
        'f_min': 0,
        'f_max': 2,
        'A': 1,
        'r': 0.5
    }

    new_ba = ba.BA(params=params)

    search_space = search.SearchSpace(n_agents=10, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_ba.run(search_space, new_function, pre_evaluate=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constant.TEST_EPSILON, 'The algorithm ba failed to converge.'
