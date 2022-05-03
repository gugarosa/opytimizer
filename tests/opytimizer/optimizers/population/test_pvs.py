import numpy as np

from opytimizer.optimizers.population import pvs
from opytimizer.spaces import search


def test_pvs_update():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_pvs = pvs.PVS()

    new_pvs.update(search_space, square)
