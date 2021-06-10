import numpy as np

from opytimizer.optimizers.population import coa
from opytimizer.spaces import search

np.random.seed(0)


def test_coa_params():
    params = {
        'n_p': 2
    }

    new_coa = coa.COA(params=params)

    assert new_coa.n_p == 2


def test_coa_params_setter():
    new_coa = coa.COA()

    try:
        new_coa.n_p = 'a'
    except:
        new_coa.n_p = 2

    assert new_coa.n_p == 2

    try:
        new_coa.n_p = -1
    except:
        new_coa.n_p = 2

    assert new_coa.n_p == 2


def test_coa_compile():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_coa = coa.COA()
    new_coa.compile(search_space)

    try:
        new_coa.n_c = 'a'
    except:
        new_coa.n_c = 1

    assert new_coa.n_c == 1

    try:
        new_coa.n_c = -1
    except:
        new_coa.n_c = 1

    assert new_coa.n_c == 1


def test_coa_get_agents_from_pack():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_coa = coa.COA()
    new_coa.compile(search_space)

    agents = new_coa._get_agents_from_pack(search_space.agents, 0)

    assert len(agents) == 5

    agents = new_coa._get_agents_from_pack(search_space.agents, 1)

    assert len(agents) == 5


def test_coa_transition_packs():
    search_space = search.SearchSpace(n_agents=200, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_coa = coa.COA()
    new_coa.compile(search_space)

    new_coa._transition_packs(search_space.agents)


def test_coa_update():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_coa = coa.COA()
    new_coa.compile(search_space)

    new_coa.update(search_space, square)
