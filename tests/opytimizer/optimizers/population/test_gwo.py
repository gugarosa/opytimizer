import numpy as np

from opytimizer.optimizers.population import gwo
from opytimizer.spaces import search

np.random.seed(0)


def test_gwo_calculate_coefficients():
    new_gwo = gwo.GWO()

    A, C = new_gwo._calculate_coefficients(1)

    assert A[0] == 0.0976270078546495
    assert C[0] == 1.430378732744839


def test_gwo_update():
    def square(x):
        return np.sum(x**2)

    new_gwo = gwo.GWO()

    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_gwo.update(search_space, square, 1, 10)
