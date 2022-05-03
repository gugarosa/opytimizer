import numpy as np

from opytimizer.optimizers.evolutionary import rra
from opytimizer.spaces import search

np.random.seed(0)


def test_rra_params():
    params = {"d_runner": 2, "d_root": 0.01, "tol": 0.01, "max_stall": 1000}

    new_rra = rra.RRA(params=params)

    assert new_rra.d_runner == 2

    assert new_rra.d_root == 0.01

    assert new_rra.tol == 0.01

    assert new_rra.max_stall == 1000


def test_rra_params_setter():
    new_rra = rra.RRA()

    try:
        new_rra.d_runner = "a"
    except:
        new_rra.d_runner = 2

    assert new_rra.d_runner == 2

    try:
        new_rra.d_runner = -1
    except:
        new_rra.d_runner = 2

    assert new_rra.d_runner == 2

    try:
        new_rra.d_root = "b"
    except:
        new_rra.d_root = 0.01

    assert new_rra.d_root == 0.01

    try:
        new_rra.d_root = -1
    except:
        new_rra.d_root = 0.01

    assert new_rra.d_root == 0.01

    try:
        new_rra.tol = "c"
    except:
        new_rra.tol = 0.01

    assert new_rra.tol == 0.01

    try:
        new_rra.tol = -1
    except:
        new_rra.tol = 0.01

    assert new_rra.tol == 0.01

    try:
        new_rra.max_stall = "d"
    except:
        new_rra.max_stall = 1000

    assert new_rra.max_stall == 1000

    try:
        new_rra.max_stall = -1
    except:
        new_rra.max_stall = 1000

    assert new_rra.max_stall == 1000

    try:
        new_rra.n_stall = "e"
    except:
        new_rra.n_stall = 0

    assert new_rra.n_stall == 0

    try:
        new_rra.n_stall = -1
    except:
        new_rra.n_stall = 0

    assert new_rra.n_stall == 0

    try:
        new_rra.n_stall = 1001
    except:
        new_rra.n_stall = 0

    assert new_rra.n_stall == 0

    try:
        new_rra.last_best_fit = "e"
    except:
        new_rra.last_best_fit = 0

    assert new_rra.last_best_fit == 0


def test_rra_stalling_search():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_rra = rra.RRA()

    new_rra._stalling_search(search_space.agents, square, is_large=True)
    new_rra._stalling_search(search_space.agents, square, is_large=False)


def test_rra_roulette_selection():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    fitness = [1 for _ in range(search_space.n_agents)]

    new_rra = rra.RRA()

    idx = new_rra._roulette_selection(fitness)

    assert idx >= 0


def test_rra_update():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_rra = rra.RRA()

    new_rra.tol = 1e10
    new_rra.max_stall = 1
    new_rra.update(search_space, square)

    new_rra.tol = 1e-500
    new_rra.update(search_space, square)
