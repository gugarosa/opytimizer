import numpy as np

from opytimizer.optimizers.science import wca
from opytimizer.spaces import search


def test_wca_params():
    params = {
        'nsr': 5,
        'd_max': 0.25
    }

    new_wca = wca.WCA(params=params)

    assert new_wca.nsr == 5

    assert new_wca.d_max == 0.25


def test_wca_params_setter():
    new_wca = wca.WCA()

    try:
        new_wca.nsr = 0.0
    except:
        new_wca.nsr = 10

    try:
        new_wca.nsr = 0
    except:
        new_wca.nsr = 10

    assert new_wca.nsr == 10

    try:
        new_wca.d_max = 'a'
    except:
        new_wca.d_max = 0.1

    try:
        new_wca.d_max = -1
    except:
        new_wca.d_max = 0.1

    assert new_wca.d_max == 0.1


def test_wca_compile():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_wca = wca.WCA()
    new_wca.compile(search_space)

    try:
        new_wca.flows = 1
    except:
        new_wca.flows = np.array([1])

    assert new_wca.flows == np.array([1])


def test_wca_flow_intensity():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_wca = wca.WCA()
    new_wca.compile(search_space)


def test_wca_raining_process():
    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_wca = wca.WCA()
    new_wca.compile(search_space)
    new_wca.flows[0] = 5

    new_wca.d_max = 100
    new_wca._raining_process(search_space.agents, search_space.best_agent)


def test_wca_update_stream():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_wca = wca.WCA()
    new_wca.compile(search_space)
    new_wca.flows[0] = 5

    new_wca._update_stream(search_space.agents, square)


def test_wca_update_river():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_wca = wca.WCA()
    new_wca.compile(search_space)

    new_wca._update_river(search_space.agents, search_space.best_agent, square)


def test_wca_update():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_wca = wca.WCA()
    new_wca.compile(search_space)

    new_wca.update(search_space, square, 1)
