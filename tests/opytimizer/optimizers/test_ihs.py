import sys

import numpy as np
import pytest
from opytimizer.core import function
from opytimizer.optimizers import ihs
from opytimizer.spaces import search


def test_ihs_hyperparams():
    hyperparams = {
        'HMCR': 0.5,
        'PAR_min': 0.5,
        'PAR_max': 1,
        'bw_min': 2,
        'bw_max': 5
    }

    new_ihs = ihs.IHS(hyperparams=hyperparams)

    assert new_ihs.HMCR == 0.5

    assert new_ihs.PAR_min == 0.5

    assert new_ihs.PAR_max == 1

    assert new_ihs.bw_min == 2

    assert new_ihs.bw_max == 5


def test_ihs_hyperparams_setter():
    new_ihs = ihs.IHS()

    new_ihs.HMCR = 0.7
    assert new_ihs.HMCR == 0.7

    new_ihs.PAR_min = 0.1
    assert new_ihs.PAR_min == 0.1

    new_ihs.PAR_max = 0.5
    assert new_ihs.PAR_max == 0.5

    new_ihs.bw_min = 1
    assert new_ihs.bw_min == 1

    new_ihs.bw_max = 10
    assert new_ihs.bw_max == 10


def test_ihs_rebuild():
    new_ihs = ihs.IHS()

    assert new_ihs.built == True


def test_ihs_run():
    def square(x):
        return np.sum(x**2)

    new_function = function.Function(pointer=square)

    new_ihs = ihs.IHS()

    search_space = search.SearchSpace(n_agents=2, n_iterations=10,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_ihs.run(search_space, new_function)

    assert len(history.history) > 0
