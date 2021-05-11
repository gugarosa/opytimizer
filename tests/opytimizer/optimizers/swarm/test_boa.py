import numpy as np

from opytimizer.optimizers.swarm import boa
from opytimizer.spaces import search


def test_boa_params():
    params = {
        'c': 0.01,
        'a': 0.1,
        'p': 0.8
    }

    new_boa = boa.BOA(params=params)

    assert new_boa.c == 0.01

    assert new_boa.a == 0.1

    assert new_boa.p == 0.8


def test_boa_params_setter():
    new_boa = boa.BOA()

    try:
        new_boa.c = 'a'
    except:
        new_boa.c = 0.01

    try:
        new_boa.c = -1
    except:
        new_boa.c = 0.01

    assert new_boa.c == 0.01

    try:
        new_boa.a = 'b'
    except:
        new_boa.a = 0.1

    try:
        new_boa.a = -1
    except:
        new_boa.a = 0.1

    assert new_boa.a == 0.1

    try:
        new_boa.p = 'c'
    except:
        new_boa.p = 0.8

    try:
        new_boa.p = -1
    except:
        new_boa.p = 0.8

    assert new_boa.p == 0.8


def test_boa_create_additional_attrs():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_boa = boa.BOA()
    new_boa.create_additional_attrs(search_space)

    try:
        new_boa.fragrance = 1
    except:
        new_boa.fragrance = np.array([1])

    assert new_boa.fragrance == np.array([1])


def test_boa_best_movement():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_boa = boa.BOA()
    new_boa.create_additional_attrs(search_space)

    new_boa._best_movement(
        search_space.agents[0].position, search_space.best_agent.position, new_boa.fragrance[0], 0.5)


def test_boa_local_movement():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_boa = boa.BOA()
    new_boa.create_additional_attrs(search_space)

    new_boa._local_movement(search_space.agents[0].position, search_space.agents[1].position,
                            search_space.agents[2].position, new_boa.fragrance[0], 0.5)


def test_boa_update():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_boa = boa.BOA()
    new_boa.create_additional_attrs(search_space)

    new_boa.update(search_space)
