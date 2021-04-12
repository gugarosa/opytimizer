import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.population import gco
from opytimizer.spaces import search
from opytimizer.utils import constants

np.random.seed(0)


def test_gco_hyperparams():
    hyperparams = {
        'CR': 0.7,
        'F': 1.25
    }

    new_gco = gco.GCO(hyperparams=hyperparams)

    assert new_gco.CR == 0.7

    assert new_gco.F == 1.25


def test_gco_hyperparams_setter():
    new_gco = gco.GCO()

    try:
        new_gco.CR = 'a'
    except:
        new_gco.CR = 0.75

    try:
        new_gco.CR = -1
    except:
        new_gco.CR = 0.75

    assert new_gco.CR == 0.75

    try:
        new_gco.F = 'b'
    except:
        new_gco.F = 1.5

    try:
        new_gco.F = -1
    except:
        new_gco.F = 1.5

    assert new_gco.F == 1.5


def test_gco_build():
    new_gco = gco.GCO()

    assert new_gco.built == True


def test_gco_mutate_cell():
    new_gco = gco.GCO()

    search_space = search.SearchSpace(n_agents=4, n_iterations=10,
                                      n_variables=2, lower_bound=[1, 1],
                                      upper_bound=[10, 10])

    cell = new_gco._mutate_cell(
        search_space.agents[0], search_space.agents[1], search_space.agents[2], search_space.agents[3])

    assert cell.position[0][0] != 0


def test_gco_update():
    def square(x):
        return np.sum(x**2)

    new_function = function.Function(pointer=square)

    new_gco = gco.GCO()

    search_space = search.SearchSpace(n_agents=4, n_iterations=10,
                                      n_variables=2, lower_bound=[1, 1],
                                      upper_bound=[10, 10])

    new_gco._update(search_space.agents, new_function, np.array([70, 70, 70, 70]), np.array([1, 1, 1, 1]))

    assert search_space.agents[0].position[0] != 0


def test_gco_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    new_gco = gco.GCO()

    search_space = search.SearchSpace(n_agents=10, n_iterations=30,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_gco.run(search_space, new_function, pre_evaluate=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, 'The algorithm gco failed to converge.'
