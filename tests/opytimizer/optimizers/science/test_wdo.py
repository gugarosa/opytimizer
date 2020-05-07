import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.science import wdo
from opytimizer.spaces import search
from opytimizer.utils import constants


def test_wdo_hyperparams():
    hyperparams = {
        'v_max': 0.3,
        'alpha': 0.8,
        'g': 0.6,
        'c': 1.0,
        'RT': 1.5
    }

    new_wdo = wdo.WDO(hyperparams=hyperparams)

    assert new_wdo.v_max == 0.3

    assert new_wdo.alpha == 0.8

    assert new_wdo.g == 0.6

    assert new_wdo.c == 1.0

    assert new_wdo.RT == 1.5


def test_wdo_hyperparams_setter():
    new_wdo = wdo.WDO()

    try:
        new_wdo.v_max = 'a'
    except:
        new_wdo.v_max = 0.1

    try:
        new_wdo.v_max = -1
    except:
        new_wdo.v_max = 0.1

    assert new_wdo.v_max == 0.1

    try:
        new_wdo.alpha = 'b'
    except:
        new_wdo.alpha = 0.8

    try:
        new_wdo.alpha = -1
    except:
        new_wdo.alpha = 0.8

    assert new_wdo.alpha == 0.8

    try:
        new_wdo.g = 'c'
    except:
        new_wdo.g = 0.5

    try:
        new_wdo.g = -1
    except:
        new_wdo.g = 0.5

    assert new_wdo.g == 0.5

    try:
        new_wdo.c = 'd'
    except:
        new_wdo.c = 0.5

    try:
        new_wdo.c = -1
    except:
        new_wdo.c = 0.5

    assert new_wdo.c == 0.5

    try:
        new_wdo.RT = 'e'
    except:
        new_wdo.RT = 0.5

    try:
        new_wdo.RT = -1
    except:
        new_wdo.RT = 0.5

    assert new_wdo.RT == 0.5


def test_wdo_build():
    new_wdo = wdo.WDO()

    assert new_wdo.built == True


def test_wdo_update_velocity():
    new_wdo = wdo.WDO()

    velocity = new_wdo._update_velocity(1, 1, 1, 1, 1)

    assert velocity != 0


def test_wdo_update_position():
    new_wdo = wdo.WDO()

    position = new_wdo._update_position(1, 1)

    assert position == 2


def test_wdo_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    hyperparams = {
        'v_max': 0.3,
        'alpha': 0.8,
        'g': 0.6,
        'c': 1.0,
        'RT': 1.5
    }

    new_wdo = wdo.WDO(hyperparams=hyperparams)

    search_space = search.SearchSpace(n_agents=10, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_wdo.run(search_space, new_function, pre_evaluation=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, 'The algorithm wdo failed to converge.'
