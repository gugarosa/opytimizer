import numpy as np

from opytimizer.optimizers.science import aig
from opytimizer.spaces import search


def test_aig_params():
    params = {"alpha": np.pi, "beta": np.pi}

    new_aig = aig.AIG(params=params)

    assert new_aig.alpha == np.pi

    assert new_aig.beta == np.pi


def test_aig_params_setter():
    new_aig = aig.AIG()

    try:
        new_aig.alpha = "a"
    except:
        new_aig.alpha = np.pi

    assert new_aig.alpha == np.pi

    try:
        new_aig.alpha = -1
    except:
        new_aig.alpha = np.pi

    assert new_aig.alpha == np.pi

    try:
        new_aig.beta = "b"
    except:
        new_aig.beta = np.pi

    assert new_aig.beta == np.pi

    try:
        new_aig.beta = -1
    except:
        new_aig.beta = np.pi

    assert new_aig.beta == np.pi


def test_aig_update():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[-10, -10], upper_bound=[10, 10]
    )

    new_aig = aig.AIG()

    new_aig.update(search_space, square)
