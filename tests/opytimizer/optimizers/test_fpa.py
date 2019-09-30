import numpy as np

from opytimizer.core import function
from opytimizer.optimizers import fpa
from opytimizer.spaces import search
from opytimizer.utils import constants


def test_fpa_hyperparams():
    hyperparams = {
        'beta': 1.0,
        'eta': 0.5,
        'p': 0.5
    }

    new_fpa = fpa.FPA(hyperparams=hyperparams)

    assert new_fpa.beta == 1.0

    assert new_fpa.eta == 0.5

    assert new_fpa.p == 0.5


def test_fpa_hyperparams_setter():
    new_fpa = fpa.FPA()

    try:
        new_fpa.beta = 'a'
    except:
        new_fpa.beta = 0.75

    try:
        new_fpa.beta = -1
    except:
        new_fpa.beta = 0.75

    assert new_fpa.beta == 0.75

    try:
        new_fpa.eta = 'b'
    except:
        new_fpa.eta = 1.5

    try:
        new_fpa.eta = -1
    except:
        new_fpa.eta = 1.5

    assert new_fpa.eta == 1.5

    try:
        new_fpa.p = 'c'
    except:
        new_fpa.p = 0.25

    try:
        new_fpa.p = -1
    except:
        new_fpa.p = 0.25

    assert new_fpa.p == 0.25


def test_fpa_build():
    new_fpa = fpa.FPA()

    assert new_fpa.built == True


def test_fpa_global_pollination():
    new_fpa = fpa.FPA()

    position = new_fpa._global_pollination(1, 2)

    assert position != 0


def test_fpa_local_pollination():
    new_fpa = fpa.FPA()

    position = new_fpa._local_pollination(1, 2, 1, 0.5)

    assert position == 1.5


def test_fpa_update():
    def square(x):
        return np.sum(x**2)

    new_function = function.Function(pointer=square)

    new_fpa = fpa.FPA()

    search_space = search.SearchSpace(n_agents=2, n_iterations=10,
                                      n_variables=2, lower_bound=[1, 1],
                                      upper_bound=[10, 10])

    new_fpa._update(search_space.agents, search_space.best_agent, new_function)

    assert search_space.agents[0].position[0] != 0


def test_fpa_run():
    def square(x):
        return np.sum(x**2)

    new_function = function.Function(pointer=square)

    new_fpa = fpa.FPA()

    search_space = search.SearchSpace(n_agents=10, n_iterations=30,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_fpa.run(search_space, new_function)

    assert len(history.agents) > 0
    assert len(history.best) > 0

    best_fitness = history.best[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, 'The algorithm fpa failed to converge.'
