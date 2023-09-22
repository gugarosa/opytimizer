import numpy as np

from opytimizer.optimizers.swarm import waoa
from opytimizer.spaces import search

def test_waoa_update():
    def square(x):
        return np.sum(x**2)

    new_waoa = waoa.WAOA()

    search_space = search.SearchSpace(
        n_agents=20, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_waoa.update(search_space, square)