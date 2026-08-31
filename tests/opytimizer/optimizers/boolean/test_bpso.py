import sys

import numpy as np

from opytimizer.optimizers.boolean import bpso
from opytimizer.spaces import boolean


def test_bpso_params():
    params = {
        "c1": np.array([1]),
        "c2": np.array([1]),
    }

    new_bpso = bpso.BPSO(params=params)

    assert new_bpso.c1 == 0 or new_bpso.c1 == 1

    assert new_bpso.c2 == 0 or new_bpso.c2 == 1


def test_bpso_compile():
    boolean_space = boolean.BooleanSpace(n_agents=2, n_variables=5)

    new_bpso = bpso.BPSO()
    new_bpso.compile(boolean_space)


def test_bpso_evaluate():
    def square(x):
        return np.sum(x**2)

    new_function = square

    boolean_space = boolean.BooleanSpace(n_agents=2, n_variables=5)

    new_bpso = bpso.BPSO()
    new_bpso.compile(boolean_space)

    new_bpso.evaluate(boolean_space, new_function)

    assert boolean_space.best_agent.fit < sys.float_info.max


def test_bpso_update():
    boolean_space = boolean.BooleanSpace(n_agents=2, n_variables=5)

    new_bpso = bpso.BPSO()
    new_bpso.compile(boolean_space)

    new_bpso.update(boolean_space)
