import numpy as np

from opytimizer.optimizers.population import loa
from opytimizer.spaces import search

np.random.seed(0)


def test_lion_params():
    new_lion = loa.Lion(1, 1, [0], [1], np.array([0]), 0)

    assert new_lion.best_position[0] == 0

    assert new_lion.p_fit == 0

    assert new_lion.nomad == False

    assert new_lion.female == False

    assert new_lion.pride == 0

    assert new_lion.group == 0


def test_lion_params_setter():
    new_lion = loa.Lion(1, 1, [0], [1], np.array([0]), 0)

    try:
        new_lion.best_position = 0
    except:
        new_lion.best_position = np.array([0])

    assert new_lion.best_position == np.array([0])

    try:
        new_lion.p_fit = 'a'
    except:
        new_lion.p_fit = 0

    assert new_lion.p_fit == 0

    try:
        new_lion.nomad = 'b'
    except:
        new_lion.nomad = False

    assert new_lion.nomad == False

    try:
        new_lion.female = 'c'
    except:
        new_lion.female = False

    assert new_lion.female == False

    try:
        new_lion.pride = 'd'
    except:
        new_lion.pride = 0

    assert new_lion.pride == 0

    try:
        new_lion.pride = -1
    except:
        new_lion.pride = 0

    assert new_lion.pride == 0

    try:
        new_lion.group = 'e'
    except:
        new_lion.group = 0

    assert new_lion.group == 0

    try:
        new_lion.group = -1
    except:
        new_lion.group = 0

    assert new_lion.group == 0
