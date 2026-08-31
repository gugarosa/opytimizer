import numpy as np

from opytimizer.optimizers.swarm import sso
from opytimizer.spaces import search


def test_sso_params():
    params = {"C_w": 0.1, "C_p": 0.4, "C_g": 0.9}

    new_sso = sso.SSO(params=params)

    assert new_sso.C_w == 0.1

    assert new_sso.C_p == 0.4

    assert new_sso.C_g == 0.9


def test_sso_compile():
    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_sso = sso.SSO()
    new_sso.compile(search_space)


def test_sso_evaluate():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_sso = sso.SSO()
    new_sso.compile(search_space)

    new_sso.evaluate(search_space, square)


def test_sso_update():
    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_sso = sso.SSO()
    new_sso.compile(search_space)

    new_sso.update(search_space)
