import numpy as np
import opytimizer.math.random as r
from opytimizer.optimizers.population import rfo
from opytimizer.spaces import search

np.random.seed(1)


def test_rfo_params():
    params = {
        'phi': r.generate_uniform_random_number(0, 2*np.pi)[0],
        'theta': r.generate_uniform_random_number()[0],
        'p_replacement': 0.05
    }

    new_rfo = rfo.RFO(params=params)

    assert 0 <= new_rfo.phi <= 2*np.pi

    assert 0 <= new_rfo.theta <= 1

    assert new_rfo.p_replacement == 0.05


def test_rfo_params_setter():
    new_rfo = rfo.RFO()

    try:
        new_rfo.phi = 'a'
    except:
        new_rfo.phi = r.generate_uniform_random_number(0, 2*np.pi)[0]

    assert 0 <= new_rfo.phi <= 2*np.pi

    try:
        new_rfo.phi = -1
    except:
        new_rfo.phi = r.generate_uniform_random_number(0, 2*np.pi)[0]

    assert 0 <= new_rfo.phi <= 2*np.pi

    try:
        new_rfo.theta = 'b'
    except:
        new_rfo.theta = r.generate_uniform_random_number()[0]

    assert 0 <= new_rfo.theta <= 1

    try:
        new_rfo.theta = -1
    except:
        new_rfo.theta = r.generate_uniform_random_number()[0]

    assert 0 <= new_rfo.theta <= 1

    try:
        new_rfo.p_replacement = 'c'
    except:
        new_rfo.p_replacement = 0.05

    assert new_rfo.p_replacement == 0.05

    try:
        new_rfo.p_replacement = -1
    except:
        new_rfo.p_replacement = 0.05

    assert new_rfo.p_replacement == 0.05


def test_rfo_compile():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_rfo = rfo.RFO()
    new_rfo.compile(search_space)

    try:
        new_rfo.n_replacement = 'a'
    except:
        new_rfo.n_replacement = 1

    assert new_rfo.n_replacement == 1

    try:
        new_rfo.n_replacement = -1
    except:
        new_rfo.n_replacement = 1

    assert new_rfo.n_replacement == 1


def test_rfo_rellocation():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_rfo = rfo.RFO()
    new_rfo.compile(search_space)

    new_rfo._rellocation(
        search_space.agents[0], search_space.best_agent, square)


def test_rfo_noticing():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_rfo = rfo.RFO()
    new_rfo.compile(search_space)

    new_rfo._noticing(search_space.agents[0], square, 0.1)

    new_rfo.phi = 0
    new_rfo._noticing(search_space.agents[0], square, 0.1)


def test_rfo_update():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_rfo = rfo.RFO()
    new_rfo.compile(search_space)

    new_rfo.update(search_space, square)

    new_rfo.n_replacement = 10
    new_rfo.update(search_space, square)
