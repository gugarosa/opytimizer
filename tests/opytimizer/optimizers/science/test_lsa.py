import numpy as np

from opytimizer.optimizers.science import lsa
from opytimizer.spaces import search


def test_lsa_params():
    params = {
        'max_time': 10,
        'E': 2.05,
        'p_fork': 0.01
    }

    new_lsa = lsa.LSA(params=params)

    assert new_lsa.max_time == 10
    assert new_lsa.E == 2.05
    assert new_lsa.p_fork == 0.01


def test_lsa_params_setter():
    new_lsa = lsa.LSA()

    try:
        new_lsa.max_time = 'a'
    except:
        new_lsa.max_time = 10

    assert new_lsa.max_time == 10

    try:
        new_lsa.max_time = -1
    except:
        new_lsa.max_time = 10

    assert new_lsa.max_time == 10

    try:
        new_lsa.E = 'b'
    except:
        new_lsa.E = 2.05

    assert new_lsa.E == 2.05

    try:
        new_lsa.E = -1
    except:
        new_lsa.E = 2.05

    assert new_lsa.E == 2.05

    try:
        new_lsa.p_fork = 'c'
    except:
        new_lsa.p_fork = 0.01

    assert new_lsa.p_fork == 0.01

    try:
        new_lsa.p_fork = -1
    except:
        new_lsa.p_fork = 0.01

    assert new_lsa.p_fork == 0.01


def test_lsa_compile():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_lsa = lsa.LSA()
    new_lsa.compile(search_space)

    try:
        new_lsa.time = 'a'
    except:
        new_lsa.time = 0

    assert new_lsa.time == 0

    try:
        new_lsa.time = -1
    except:
        new_lsa.time = 0

    assert new_lsa.time == 0

    try:
        new_lsa.direction = 1
    except:
        new_lsa.direction = np.array([1])

    assert new_lsa.direction == np.array([1])


def test_lsa_update_direction():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_lsa = lsa.LSA()
    new_lsa.compile(search_space)

    new_lsa._update_direction(search_space.agents[0], square)


def test_lsa_update_position():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_lsa = lsa.LSA()
    new_lsa.compile(search_space)

    new_lsa._update_position(
        search_space.agents[0], search_space.agents[0], square, 0.5)


def test_lsa_update():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_lsa = lsa.LSA()
    new_lsa.compile(search_space)

    new_lsa.p_fork = 1
    new_lsa.update(search_space, square, 1, 10)

    new_lsa.time = 11
    new_lsa.update(search_space, square, 1, 10)
