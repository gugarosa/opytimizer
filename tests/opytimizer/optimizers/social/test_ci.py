import numpy as np

from opytimizer.optimizers.social import ci
from opytimizer.spaces import search


def test_ci_params():
    params = {
        'r': 0.8,
        't': 3
    }

    new_ci = ci.CI(params=params)

    assert new_ci.r == 0.8

    assert new_ci.t == 3


def test_ci_params_setter():
    new_ci = ci.CI()

    try:
        new_ci.r = 'a'
    except:
        new_ci.r = 0.8

    assert new_ci.r == 0.8

    try:
        new_ci.r = -1
    except:
        new_ci.r = 0.8

    assert new_ci.r == 0.8

    try:
        new_ci.t = 'b'
    except:
        new_ci.t = 3

    assert new_ci.t == 3

    try:
        new_ci.t = -1
    except:
        new_ci.t = 3

    assert new_ci.t == 3


def test_ci_compile():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_ci = ci.CI()
    new_ci.compile(search_space)

    try:
        new_ci.lower = 1
    except:
        new_ci.lower = np.array([1])

    assert new_ci.lower == 1

    try:
        new_ci.upper = 1
    except:
        new_ci.upper = np.array([1])

    assert new_ci.upper == 1


def test_ci_update():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_ci = ci.CI()
    new_ci.compile(search_space)

    new_ci.evaluate(search_space, square)
    new_ci.update(search_space, square)
