import sys

import numpy as np
from opytimark.markers.boolean import Knapsack

import opytimizer.math.random as r
from opytimizer.core import function
from opytimizer.optimizers.boolean import bpso
from opytimizer.spaces import boolean
from opytimizer.utils import constant

np.random.seed(0)


def test_bpso_params():
    params = {
        'c1': r.generate_binary_random_number(size=(1, 1)),
        'c2': r.generate_binary_random_number(size=(1, 1))
    }

    new_bpso = bpso.BPSO(params=params)

    assert new_bpso.c1 == 0 or new_bpso.c1 == 1

    assert new_bpso.c2 == 0 or new_bpso.c2 == 1


def test_bpso_params_setter():
    new_bpso = bpso.BPSO()

    try:
        new_bpso.c1 = 'a'
    except:
        new_bpso.c1 = r.generate_binary_random_number(size=(1, 1))

    assert new_bpso.c1 == 0 or new_bpso.c1 == 1

    try:
        new_bpso.c2 = 'b'
    except:
        new_bpso.c2 = r.generate_binary_random_number(size=(1, 1))

    assert new_bpso.c2 == 0 or new_bpso.c2 == 1


def test_bpso_build():
    new_bpso = bpso.BPSO()

    assert new_bpso.built == True


def test_bpso_update_velocity():
    new_bpso = bpso.BPSO()

    velocity = new_bpso._update_velocity(
        np.array([1]), np.array([1]), np.array([1]))

    assert velocity == 0 or velocity == 1


def test_bpso_update_position():
    new_bpso = bpso.BPSO()

    position = new_bpso._update_position(1, 1)

    assert position == 0 or position == 1


def test_bpso_evaluate():
    def square(x):
        return np.sum(x**2)

    new_function = function.Function(pointer=square)

    boolean_space = boolean.BooleanSpace(
        n_agents=2, n_iterations=10, n_variables=2)

    new_bpso = bpso.BPSO()

    local_position = np.zeros((2, 2, 1))

    new_bpso._evaluate(boolean_space, new_function, local_position)

    assert boolean_space.best_agent.fit < sys.float_info.max


def test_bpso_run():
    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=Knapsack(
        values=(55, 10, 47, 5, 4), weights=(95, 4, 60, 32, 23), max_capacity=100))

    new_bpso = bpso.BPSO()

    boolean_space = boolean.BooleanSpace(
        n_agents=2, n_iterations=10, n_variables=5)

    history = new_bpso.run(boolean_space, new_function, pre_evaluate=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0
    assert len(history.local) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constant.TEST_EPSILON, 'The algorithm bpso failed to converge.'
