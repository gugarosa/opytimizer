import sys

import numpy as np

import opytimizer.math.random as r
from opytimizer.core import function
from opytimizer.optimizers.boolean import bpso
from opytimizer.spaces import boolean


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


def test_bpso_create_additional_attrs():
    boolean_space = boolean.BooleanSpace(n_agents=2, n_variables=5)
    
    new_bpso = bpso.BPSO()
    new_bpso.create_additional_attrs(boolean_space)

    try:
        new_bpso.local_position = 1
    except:
        new_bpso.local_position = np.array([1])

    assert new_bpso.local_position == 1

    try:
        new_bpso.velocity = 1
    except:
        new_bpso.velocity = np.array([1])

    assert new_bpso.velocity == 1


def test_bpso_evaluate():
    def square(x):
        return np.sum(x**2)

    new_function = function.Function(pointer=square)

    boolean_space = boolean.BooleanSpace(n_agents=2, n_variables=5)

    new_bpso = bpso.BPSO()
    new_bpso.create_additional_attrs(boolean_space)

    new_bpso.evaluate(boolean_space, new_function)

    assert boolean_space.best_agent.fit < sys.float_info.max


def test_bpso_update():
    boolean_space = boolean.BooleanSpace(n_agents=2, n_variables=5)

    new_bpso = bpso.BPSO()
    new_bpso.create_additional_attrs(boolean_space)

    new_bpso.update(boolean_space)
