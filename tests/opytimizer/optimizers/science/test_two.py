import numpy as np

from opytimizer.optimizers.science import two
from opytimizer.spaces import search


def test_two_params():
    params = {
        'mu_s': 1,
        'mu_k': 1,
        'delta_t': 1,
        'alpha': 0.9,
        'beta': 0.05
    }

    new_two = two.TWO(params=params)

    assert new_two.mu_s == 1

    assert new_two.mu_k == 1

    assert new_two.delta_t == 1

    assert new_two.alpha == 0.9

    assert new_two.beta == 0.05


def test_two_params_setter():
    new_two = two.TWO()

    try:
        new_two.mu_s = 'a'
    except:
        new_two.mu_s = 1

    try:
        new_two.mu_s = -1
    except:
        new_two.mu_s = 1

    assert new_two.mu_s == 1

    try:
        new_two.mu_k = 'b'
    except:
        new_two.mu_k = 1

    try:
        new_two.mu_k = -1
    except:
        new_two.mu_k = 1

    assert new_two.mu_k == 1

    try:
        new_two.delta_t = 'c'
    except:
        new_two.delta_t = 1

    try:
        new_two.delta_t = -1
    except:
        new_two.delta_t = 1

    assert new_two.delta_t == 1

    try:
        new_two.alpha = 'd'
    except:
        new_two.alpha = 0.9

    try:
        new_two.alpha = 0.89
    except:
        new_two.alpha = 0.9

    assert new_two.alpha == 0.9

    try:
        new_two.beta = 'e'
    except:
        new_two.beta = 0.05

    try:
        new_two.beta = -1
    except:
        new_two.beta = 0.05

    assert new_two.beta == 0.05


def test_two_constraint_handle():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_two = two.TWO()

    new_two._constraint_handle(
        search_space.agents, search_space.best_agent, square, 1)


def test_two_update():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_two = two.TWO()

    new_two.update(search_space, square, 1, 10)
    new_two.update(search_space, square, 5, 10)
