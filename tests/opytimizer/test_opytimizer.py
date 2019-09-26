import numpy as np
import pytest

import opytimizer
from opytimizer.core import function
from opytimizer.optimizers import pso
from opytimizer.spaces import search


def test_opytimizer_build():
    def square(x):
        return np.sum(x**2)

    assert square(2) == 4

    new_function = function.Function(pointer=square)

    lb = [0]

    ub = [10]

    new_space = search.SearchSpace(lower_bound=lb, upper_bound=ub)

    new_pso = pso.PSO()

    try:
        new_pso.built = False
        new_opytimizer = opytimizer.Opytimizer(
            space=new_space, optimizer=new_pso, function=new_function)
    except:
        new_pso.built = True
        new_opytimizer = opytimizer.Opytimizer(
            space=new_space, optimizer=new_pso, function=new_function)


def test_opytimizer_start():
    def square(x):
        return np.sum(x**2)

    new_function = function.Function(pointer=square)

    lb = [0]

    ub = [10]

    new_space = search.SearchSpace(lower_bound=lb, upper_bound=ub)

    new_pso = pso.PSO()

    new_opytimizer = opytimizer.Opytimizer(
        space=new_space, optimizer=new_pso, function=new_function)

    history = new_opytimizer.start()
    assert isinstance(history, opytimizer.utils.history.History)
