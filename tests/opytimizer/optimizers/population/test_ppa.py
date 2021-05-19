import numpy as np

from opytimizer.optimizers.population import ppa
from opytimizer.spaces import search


def test_ppa_compile():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_ppa = ppa.PPA()
    new_ppa.compile(search_space)

    try:
        new_ppa.velocity = 1
    except:
        new_ppa.velocity = np.array([1])

    assert new_ppa.velocity == np.array([1])


def test_ppa_calculate_population():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_ppa = ppa.PPA()
    new_ppa.compile(search_space)

    n_crows, n_cats, n_cuckoos = new_ppa._calculate_population(20, 1, 10)

    assert n_crows == 13
    assert n_cats == 1
    assert n_cuckoos == 6


def test_ppa_nesting_phase():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_ppa = ppa.PPA()
    new_ppa.compile(search_space)

    new_ppa._nesting_phase(search_space, 10)


def test_ppa_parasitism_phase():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_ppa = ppa.PPA()
    new_ppa.compile(search_space)

    new_ppa._parasitism_phase(search_space, 5, 5, 1, 10)


def test_ppa_predation_phase():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_ppa = ppa.PPA()
    new_ppa.compile(search_space)

    new_ppa._predation_phase(search_space, 5, 5, 10, 1, 10)


def test_ppa_update():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_ppa = ppa.PPA()
    new_ppa.compile(search_space)

    new_ppa.update(search_space, 1, 10)
    new_ppa.update(search_space, 9, 10)
