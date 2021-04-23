import numpy as np
from opytimark.markers.boolean import Knapsack

from opytimizer.core import function
from opytimizer.optimizers.boolean import umda
from opytimizer.spaces import boolean
from opytimizer.utils import constants

np.random.seed(0)


def test_umda_params():
    params = {
        'p_selection': 0.75,
        'lower_bound': 0.05,
        'upper_bound': 0.95
    }

    new_umda = umda.UMDA(params=params)

    assert new_umda.p_selection == 0.75

    assert new_umda.lower_bound == 0.05

    assert new_umda.upper_bound == 0.95


def test_umda_params_setter():
    new_umda = umda.UMDA()

    try:
        new_umda.p_selection = 'a'
    except:
        new_umda.p_selection = 0.75

    assert new_umda.p_selection == 0.75

    try:
        new_umda.p_selection = -1
    except:
        new_umda.p_selection = 0.75

    assert new_umda.p_selection == 0.75

    try:
        new_umda.lower_bound = 'a'
    except:
        new_umda.lower_bound = 0.05

    assert new_umda.lower_bound == 0.05

    try:
        new_umda.lower_bound = -1
    except:
        new_umda.lower_bound = 0.05

    assert new_umda.lower_bound == 0.05

    try:
        new_umda.upper_bound = 'a'
    except:
        new_umda.upper_bound = 0.95

    assert new_umda.upper_bound == 0.95

    try:
        new_umda.upper_bound = -1
    except:
        new_umda.upper_bound = 0.95

    assert new_umda.upper_bound == 0.95

    try:
        new_umda.upper_bound = 0.04
    except:
        new_umda.upper_bound = 0.95

    assert new_umda.upper_bound == 0.95


def test_umda_build():
    new_umda = umda.UMDA()

    assert new_umda.built == True


def test_umda_calculate_probability():
    new_umda = umda.UMDA()

    boolean_space = boolean.BooleanSpace(
        n_agents=5, n_iterations=20, n_variables=2)

    probs = new_umda._calculate_probability(boolean_space.agents)

    assert probs.shape == (2, 1)


def test_umda_sample_position():
    new_umda = umda.UMDA()

    probs = np.zeros((1, 1))

    position = new_umda._sample_position(probs)

    assert position == 1


def test_umda_run():
    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=Knapsack(
        values=(55, 10, 47, 5, 4), weights=(95, 4, 60, 32, 23), max_capacity=100))

    new_umda = umda.UMDA()

    boolean_space = boolean.BooleanSpace(
        n_agents=2, n_iterations=10, n_variables=5)

    history = new_umda.run(boolean_space, new_function, pre_evaluate=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, 'The algorithm umda failed to converge.'
