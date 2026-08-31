import numpy as np

from opytimizer.optimizers.misc import hc
from opytimizer.spaces import search

np.random.seed(0)


def test_hc_params():
    params = {
        "r_mean": 0,
        "r_var": 0.1,
    }

    new_hc = hc.HC(params=params)

    assert new_hc.r_mean == 0

    assert new_hc.r_var == 0.1


def test_hc_update():
    search_space = search.SearchSpace(
        n_agents=50, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_hc = hc.HC()

    new_hc.update(search_space)
