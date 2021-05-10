import numpy as np

from opytimizer.optimizers.evolutionary import bsa
from opytimizer.spaces import search

np.random.seed(0)


def test_bsa_params():
    params = {
        'F': 3.0,
        'mix_rate': 1
    }

    new_bsa = bsa.BSA(params=params)

    assert new_bsa.F == 3.0

    assert new_bsa.mix_rate == 1


def test_bsa_params_setter():
    new_bsa = bsa.BSA()

    try:
        new_bsa.F = 'a'
    except:
        new_bsa.F = 3.0

    try:
        new_bsa.mix_rate = 'b'
    except:
        new_bsa.mix_rate = 1

    try:
        new_bsa.mix_rate = -1
    except:
        new_bsa.mix_rate = 1

    assert new_bsa.mix_rate == 1


def test_bsa_create_additional_attrs():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_bsa = bsa.BSA()
    new_bsa.create_additional_attrs(search_space)

    try:
        new_bsa.old_agents = 1
    except:
        new_bsa.old_agents = []

    assert new_bsa.old_agents == []


def test_bsa_permute():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_bsa = bsa.BSA()
    new_bsa.create_additional_attrs(search_space)

    new_bsa._permute(search_space.agents)


def test_bsa_mutate():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_bsa = bsa.BSA()
    new_bsa.create_additional_attrs(search_space)

    trial_agents = new_bsa._mutate(search_space.agents)

    assert len(trial_agents) == 10


def test_bsa_crossover():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_bsa = bsa.BSA()
    new_bsa.create_additional_attrs(search_space)

    trial_agents = new_bsa._mutate(search_space.agents)
    new_bsa._crossover(search_space.agents, trial_agents)


def test_bsa_run():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=75, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_bsa = bsa.BSA()
    new_bsa.create_additional_attrs(search_space)

    new_bsa.update(search_space, square)
