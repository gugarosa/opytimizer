import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.evolutionary import iwo
from opytimizer.spaces import search
from opytimizer.utils import constant

np.random.seed(0)


def test_iwo_params():
    params = {
        'min_seeds': 0,
        'max_seeds': 5,
        'e': 2,
        'final_sigma': 0.001,
        'init_sigma': 3
    }

    new_iwo = iwo.IWO(params=params)

    assert new_iwo.min_seeds == 0

    assert new_iwo.max_seeds == 5

    assert new_iwo.e == 2

    assert new_iwo.final_sigma == 0.001

    assert new_iwo.init_sigma == 3


def test_iwo_params_setter():
    new_iwo = iwo.IWO()

    try:
        new_iwo.min_seeds = 'a'
    except:
        new_iwo.min_seeds = 0

    try:
        new_iwo.min_seeds = -1
    except:
        new_iwo.min_seeds = 0

    assert new_iwo.min_seeds == 0

    try:
        new_iwo.max_seeds = 'b'
    except:
        new_iwo.max_seeds = 2

    try:
        new_iwo.max_seeds = -1
    except:
        new_iwo.max_seeds = 2

    assert new_iwo.max_seeds == 2

    try:
        new_iwo.e = 'c'
    except:
        new_iwo.e = 1.5

    try:
        new_iwo.e = -1
    except:
        new_iwo.e = 1.5

    assert new_iwo.e == 1.5

    try:
        new_iwo.final_sigma = 'd'
    except:
        new_iwo.final_sigma = 1.5

    try:
        new_iwo.final_sigma = -1
    except:
        new_iwo.final_sigma = 1.5

    assert new_iwo.final_sigma == 1.5

    try:
        new_iwo.init_sigma = 'e'
    except:
        new_iwo.init_sigma = 2.0

    try:
        new_iwo.init_sigma = -1
    except:
        new_iwo.init_sigma = 2.0

    try:
        new_iwo.init_sigma = 1.3
    except:
        new_iwo.init_sigma = 2.0

    assert new_iwo.init_sigma == 2.0

    try:
        new_iwo.sigma = 'f'
    except:
        new_iwo.sigma = 1

    assert new_iwo.sigma == 1


def test_iwo_spatial_dispersal():
    new_iwo = iwo.IWO()

    new_iwo._spatial_dispersal(1, 10)

    assert new_iwo.sigma == 2.43019


def test_iwo_produce_offspring():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=2, n_variables=2,
                                      lower_bound=[1, 1], upper_bound=[10, 10])

    new_iwo = iwo.IWO()

    agent = new_iwo._produce_offspring(search_space.agents[0], square)

    assert type(agent).__name__ == 'Agent'


def test_iwo_update():
    def square(x):
        return np.sum(x**2)

    new_iwo = iwo.IWO()

    search_space = search.SearchSpace(n_agents=5, n_variables=2,
                lower_bound=[1, 1], upper_bound=[10, 10])

    new_iwo.update(search_space, function, 1, 10)
