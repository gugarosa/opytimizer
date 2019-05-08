import sys

import numpy as np
import pytest

from opytimizer.core import function
from opytimizer.optimizers import cs
from opytimizer.spaces import search


def test_cs_hyperparams():
    hyperparams = {
        'alpha': 1.0,
        'beta': 1.5,
        'p': 0.2
    }

    new_cs = cs.CS(hyperparams=hyperparams)

    assert new_cs.alpha == 1.0

    assert new_cs.beta == 1.5

    assert new_cs.p == 0.2


def test_cs_hyperparams_setter():
    new_cs = cs.CS()

    new_cs.alpha = 0.001
    assert new_cs.alpha == 0.001

    new_cs.beta = 0.75
    assert new_cs.beta == 0.75

    new_cs.p = 0.25
    assert new_cs.p == 0.25


def test_cs_build():
    new_cs = cs.CS()

    assert new_cs.built == True


def test_cs_update():
    def square(x):
        return np.sum(x**2)

    new_function = function.Function(pointer=square)

    new_cs = cs.CS()

    search_space = search.SearchSpace(n_agents=20, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    new_cs._update(search_space.agents, search_space.best_agent, new_function)

    assert search_space.agents[0].position[0] != 0


def test_cs_run():
    def square(x):
        return np.sum(x**2)

    new_function = function.Function(pointer=square)

    new_cs = cs.CS()

    search_space = search.SearchSpace(n_agents=2, n_iterations=10,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_cs.run(search_space, new_function)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0
