import numpy as np

from opytimizer.optimizers.swarm import ffoa
from opytimizer.spaces import search


def test_ffoa_compile():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_ffoa = ffoa.FFOA()
    new_ffoa.compile(search_space)

    try:
        new_ffoa.x_axis = 1
    except:
        new_ffoa.x_axis = []

    assert new_ffoa.x_axis == []

    try:
        new_ffoa.y_axis = 1
    except:
        new_ffoa.y_axis = []

    assert new_ffoa.y_axis == []


def test_ffoa_update():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_ffoa = ffoa.FFOA()
    new_ffoa.compile(search_space)

    new_ffoa.update(search_space, square)
