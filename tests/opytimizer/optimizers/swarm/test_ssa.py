import numpy as np

from opytimizer.optimizers.swarm import ssa
from opytimizer.spaces import search

np.random.seed(0)


def test_ssa_update():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_ssa = ssa.SSA()

    new_ssa.update(search_space, 1, 10)
    new_ssa.update(search_space, 5, 10)
    new_ssa.update(search_space, 10, 10)
