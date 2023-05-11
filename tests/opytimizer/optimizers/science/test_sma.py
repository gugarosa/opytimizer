import numpy as np

from opytimizer.optimizers.science import sma
from opytimizer.spaces import search

def test_sma_params():
    params = {"z": 0.03}

    new_sma = sma.SMA(params=params)

    assert new_sma.z == 0.03

def test_sma_params_setter():
    new_sma = sma.SMA()

    try:
        new_sma.z = "a"
    except:
        new_sma.z = 0.05
    
    try:
        new_sma.z = -1
    except:
        new_sma.z = 0.05
    
    assert new_sma.z == 0.05

def test_sma_compile():
    search_space = search.SearchSpace(
        n_agents=2, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_sma = sma.SMA()
    new_sma.compile(search_space)

    try:
        new_sma.weight = 1
    except:
        new_sma.weight = np.array([1])

    assert new_sma.weight == np.array([1])

def test_sma_update_weight():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_sma = sma.SMA()
    new_sma.compile(search_space)

    new_sma._update_weight(search_space.agents)

def test_sma_update():
    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_sma = sma.SMA()
    new_sma.compile(search_space)

    new_sma.update(search_space, 1, 10)