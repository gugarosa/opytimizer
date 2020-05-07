import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.science import efo
from opytimizer.spaces import search
from opytimizer.utils import constants


def test_efo_hyperparams():
    hyperparams = {
        'positive_field': 0.1,
        'negative_field': 0.5,
        'ps_ratio': 0.1,
        'r_ratio': 0.4
    }

    new_efo = efo.EFO(hyperparams=hyperparams)

    assert new_efo.positive_field == 0.1

    assert new_efo.negative_field == 0.5

    assert new_efo.ps_ratio == 0.1

    assert new_efo.r_ratio == 0.4


def test_efo_hyperparams_setter():
    new_efo = efo.EFO()

    try:
        new_efo.positive_field = 'a'
    except:
        new_efo.positive_field = 0.5

    try:
        new_efo.positive_field = -1
    except:
        new_efo.positive_field = 0.5

    assert new_efo.positive_field == 0.5

    try:
        new_efo.negative_field = 'b'
    except:
        new_efo.negative_field = 0.2

    try:
        new_efo.negative_field = 0.99
    except:
        new_efo.negative_field = 0.2

    try:
        new_efo.negative_field = -1
    except:
        new_efo.negative_field = 0.2

    assert new_efo.negative_field == 0.2

    try:
        new_efo.ps_ratio = 'c'
    except:
        new_efo.ps_ratio = 0.25

    try:
        new_efo.ps_ratio = -1
    except:
        new_efo.ps_ratio = 0.25

    assert new_efo.ps_ratio == 0.25

    try:
        new_efo.r_ratio = 'd'
    except:
        new_efo.r_ratio = 0.25

    try:
        new_efo.r_ratio = -1
    except:
        new_efo.r_ratio = 0.25

    assert new_efo.r_ratio == 0.25


def test_efo_build():
    new_efo = efo.EFO()

    assert new_efo.built == True


def test_efo_calculate_indexes():
    new_efo = efo.EFO()

    a, b, c = new_efo._calculate_indexes(30)

    assert a >= 0
    assert b >= 0
    assert c >= 0


def test_efo_update():
    def square(x):
        return np.sum(x**2)

    new_function = function.Function(pointer=square)

    new_efo = efo.EFO()

    search_space = search.SearchSpace(n_agents=2, n_iterations=10,
                                      n_variables=2, lower_bound=[1, 1],
                                      upper_bound=[10, 10])

    new_efo._update(search_space.agents, new_function, 1, 1)

    assert search_space.agents[0].position[0] != 0


def test_efo_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    new_efo = efo.EFO()

    search_space = search.SearchSpace(n_agents=10, n_iterations=30,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_efo.run(search_space, new_function, pre_evaluation=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, 'The algorithm efo failed to converge.'
