import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.evolutionary import ga
from opytimizer.spaces import search
from opytimizer.utils import constants

np.random.seed(0)


def test_ga_hyperparams():
    hyperparams = {
        'p_selection': 0.75,
        'p_mutation': 0.25,
        'p_crossover': 0.5,
    }

    new_ga = ga.GA(hyperparams=hyperparams)

    assert new_ga.p_selection == 0.75

    assert new_ga.p_mutation == 0.25

    assert new_ga.p_crossover == 0.5


def test_ga_hyperparams_setter():
    new_ga = ga.GA()

    try:
        new_ga.p_selection = 'a'
    except:
        new_ga.p_selection = 0.75

    try:
        new_ga.p_selection = -1
    except:
        new_ga.p_selection = 0.75

    assert new_ga.p_selection == 0.75

    try:
        new_ga.p_mutation = 'b'
    except:
        new_ga.p_mutation = 0.25

    try:
        new_ga.p_mutation = -1
    except:
        new_ga.p_mutation = 0.25

    assert new_ga.p_mutation == 0.25

    try:
        new_ga.p_crossover = 'c'
    except:
        new_ga.p_crossover = 0.5

    try:
        new_ga.p_crossover = -1
    except:
        new_ga.p_crossover = 0.5

    assert new_ga.p_crossover == 0.5


def test_ga_build():
    new_ga = ga.GA()

    assert new_ga.built == True


def test_ga_update():
    def square(x):
        return np.sum(x**2)

    new_function = function.Function(pointer=square)

    new_ga = ga.GA()

    search_space = search.SearchSpace(n_agents=10, n_iterations=10,
                                      n_variables=2, lower_bound=[1, 1],
                                      upper_bound=[10, 10])

    new_ga._evaluate(search_space, new_function)

    new_ga._update(search_space.agents, new_function)

    assert search_space.agents[0].position[0] != 0


def test_ga_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    new_ga = ga.GA()

    search_space = search.SearchSpace(n_agents=10, n_iterations=30,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_ga.run(search_space, new_function, pre_evaluate=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, 'The algorithm ga failed to converge.'
