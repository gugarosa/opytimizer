import numpy as np

from opytimizer.optimizers.evolutionary import ga
from opytimizer.spaces import search


def test_ga_params():
    params = {
        'p_selection': 0.75,
        'p_mutation': 0.25,
        'p_crossover': 0.5,
    }

    new_ga = ga.GA(params=params)

    assert new_ga.p_selection == 0.75

    assert new_ga.p_mutation == 0.25

    assert new_ga.p_crossover == 0.5


def test_ga_params_setter():
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


def test_ga_roulette_selection():
    new_ga = ga.GA()

    fitness = [10, 20, 30, 40, 50]

    idx = new_ga._roulette_selection(len(fitness), fitness)

    assert len(idx) == 4


def test_ga_crossover():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[1, 1], upper_bound=[10, 10])

    new_ga = ga.GA()

    alpha, beta = new_ga._crossover(
        search_space.agents[0], search_space.agents[1])

    assert type(alpha).__name__ == 'Agent'
    assert type(beta).__name__ == 'Agent'


def test_ga_mutation():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[1, 1], upper_bound=[10, 10])

    new_ga = ga.GA()

    alpha, beta = new_ga._mutation(
        search_space.agents[0], search_space.agents[1])

    assert type(alpha).__name__ == 'Agent'
    assert type(beta).__name__ == 'Agent'


def test_ga_update():
    def square(x):
        return np.sum(x**2)

    new_ga = ga.GA()

    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[1, 1], upper_bound=[10, 10])

    new_ga.update(search_space, square)
