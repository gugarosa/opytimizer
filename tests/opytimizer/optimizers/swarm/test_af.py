import numpy as np

from opytimizer.optimizers.swarm import af
from opytimizer.spaces import search


def test_af_params():
    params = {"c1": 0.75, "c2": 1.25, "m": 10, "Q": 0.75}

    new_af = af.AF(params=params)

    assert new_af.c1 == 0.75

    assert new_af.c2 == 1.25

    assert new_af.m == 10

    assert new_af.Q == 0.75


def test_af_compile():
    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_af = af.AF()
    new_af.compile(search_space)


def test_af_update():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_af = af.AF()
    new_af.compile(search_space)

    new_af.evaluate(search_space, square)
    new_af.update(search_space, square)
