import numpy as np

from opytimizer.core import function
from opytimizer.math import constants
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

    try:
        new_cs.alpha = 'a'
    except:
        new_cs.alpha = 0.001

    try:
        new_cs.alpha = -1
    except:
        new_cs.alpha = 0.001

    assert new_cs.alpha == 0.001

    try:
        new_cs.beta = 'b'
    except:
        new_cs.beta = 0.75

    try:
        new_cs.beta = -1
    except:
        new_cs.beta = 0.75

    assert new_cs.beta == 0.75

    try:
        new_cs.p = 'c'
    except:
        new_cs.p = 0.25

    try:
        new_cs.p = -1
    except:
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
                                      n_variables=2, lower_bound=[-10, -10],
                                      upper_bound=[10, 10])

    new_cs._update(search_space.agents, search_space.best_agent, new_function)

    assert search_space.agents[0].position[0] != 0


def test_cs_run():
    def square(x):
        return np.sum(x**2)

    new_function = function.Function(pointer=square)

    new_cs = cs.CS()

    search_space = search.SearchSpace(n_agents=25, n_iterations=30,
                                      n_variables=2, lower_bound=[-10, -10],
                                      upper_bound=[10, 10])

    history = new_cs.run(search_space, new_function)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, "The algorithm abc failed to converge"
