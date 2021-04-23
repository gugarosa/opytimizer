import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.social import ssd
from opytimizer.spaces import search
from opytimizer.utils import constant

np.random.seed(0)


def test_ssd_params():
    params = {
        'c': 2.0,
        'decay': 0.99
    }

    new_ssd = ssd.SSD(params=params)

    assert new_ssd.c == 2.0

    assert new_ssd.decay == 0.99


def test_ssd_params_setter():
    new_ssd = ssd.SSD()

    try:
        new_ssd.c = 'a'
    except:
        new_ssd.c = 0.5

    try:
        new_ssd.c = -1
    except:
        new_ssd.c = 0.5

    assert new_ssd.c == 0.5

    try:
        new_ssd.decay = 'b'
    except:
        new_ssd.decay = 0.99

    try:
        new_ssd.decay = -1
    except:
        new_ssd.decay = 0.99

    assert new_ssd.decay == 0.99


def test_ssd_build():
    new_ssd = ssd.SSD()

    assert new_ssd.built == True


def test_ssd_mean_global_solution():
    new_ssd = ssd.SSD()

    mean = new_ssd._mean_global_solution(1, 2, 3)

    assert mean != 0


def test_ssd_update_velocity():
    new_ssd = ssd.SSD()

    velocity = new_ssd._update_velocity(0.5, 10, 25)

    assert velocity[0] != 0


def test_ssd_update_position():
    new_ssd = ssd.SSD()

    position = new_ssd._update_position(1, 1)

    assert position == 2


def test_ssd_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    params = {
        'c': 2.0,
        'decay': 0.3
    }

    new_ssd = ssd.SSD(params=params)

    search_space = search.SearchSpace(n_agents=50, n_iterations=350,
                                      n_variables=2, lower_bound=[-100, -100],
                                      upper_bound=[100, 100])

    history = new_ssd.run(search_space, new_function, pre_evaluate=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constant.TEST_EPSILON, 'The algorithm ssd failed to converge.'
