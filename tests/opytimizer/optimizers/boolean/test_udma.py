import numpy as np

import opytimizer.math.random as r
from opytimizer.core import function
from opytimizer.optimizers.boolean import udma
from opytimizer.spaces import boolean
from opytimizer.utils import constants


def test_udma_hyperparams():
    hyperparams = {
        'p_selection': 0.75,
        'lower_bound': 0.05,
        'upper_bound': 0.95
    }

    new_udma = udma.UDMA(hyperparams=hyperparams)

    assert new_udma.p_selection == 0.75

    assert new_udma.lower_bound == 0.05

    assert new_udma.upper_bound == 0.95


def test_udma_hyperparams_setter():
    new_udma = udma.UDMA()

    try:
        new_udma.p_selection = 'a'
    except:
        new_udma.p_selection = 0.75

    assert new_udma.p_selection == 0.75

    try:
        new_udma.p_selection = -1
    except:
        new_udma.p_selection = 0.75

    assert new_udma.p_selection == 0.75

    try:
        new_udma.lower_bound = 'a'
    except:
        new_udma.lower_bound = 0.05

    assert new_udma.lower_bound == 0.05

    try:
        new_udma.lower_bound = -1
    except:
        new_udma.lower_bound = 0.05

    assert new_udma.lower_bound == 0.05

    try:
        new_udma.upper_bound = 'a'
    except:
        new_udma.upper_bound = 0.95

    assert new_udma.upper_bound == 0.95

    try:
        new_udma.upper_bound = -1
    except:
        new_udma.upper_bound = 0.95

    assert new_udma.upper_bound == 0.95

    try:
        new_udma.upper_bound = 0.04
    except:
        new_udma.upper_bound = 0.95

    assert new_udma.upper_bound == 0.95


def test_udma_build():
    new_udma = udma.UDMA()

    assert new_udma.built == True


def test_udma_calculate_probability():
    new_udma = udma.UDMA()

    boolean_space = boolean.BooleanSpace(
        n_agents=5, n_iterations=20, n_variables=2)

    probs = new_udma._calculate_probability(boolean_space.agents)

    assert probs.shape == (2, 1)


def test_udma_sample_position():
    new_udma = udma.UDMA()

    probs = np.zeros((1, 1))

    position = new_udma._sample_position(probs)

    assert position == 1


def test_udma_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    new_udma = udma.UDMA()

    boolean_space = boolean.BooleanSpace(
        n_agents=5, n_iterations=20, n_variables=2)

    history = new_udma.run(boolean_space, new_function, pre_evaluation=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, 'The algorithm udma failed to converge.'
