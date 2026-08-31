import numpy as np

from opytimizer.optimizers.science import esa
from opytimizer.spaces import search


def test_esa_params():
    params = {"n_electrons": 5}

    new_esa = esa.ESA(params=params)

    assert new_esa.n_electrons == 5


def test_esa_compile():
    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[1, 1], upper_bound=[10, 10]
    )

    new_esa = esa.ESA()
    new_esa.compile(search_space)


def test_esa_update():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[1, 1], upper_bound=[10, 10]
    )

    new_esa = esa.ESA()
    new_esa.compile(search_space)

    new_esa.update(search_space, square)
