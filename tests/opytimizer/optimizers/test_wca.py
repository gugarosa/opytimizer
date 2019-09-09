import numpy as np

from opytimizer.core import function
from opytimizer.math import constants
from opytimizer.optimizers import wca
from opytimizer.spaces import search


def test_wca_hyperparams():
    hyperparams = {
        'nsr': 5,
        'd_max': 0.25
    }

    new_wca = wca.WCA(hyperparams=hyperparams)

    assert new_wca.nsr == 5

    assert new_wca.d_max == 0.25


def test_wca_hyperparams_setter():
    new_wca = wca.WCA()

    new_wca.nsr = 10
    assert new_wca.nsr == 10

    new_wca.d_max = 0.1
    assert new_wca.d_max == 0.1


def test_wca_build():
    new_wca = wca.WCA()

    assert new_wca.built == True


def test_wca_flow_intensity():
    new_wca = wca.WCA()

    search_space = search.SearchSpace(n_agents=2, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    flows = new_wca._flow_intensity(search_space.agents)

    assert flows.shape[0] == new_wca.nsr


def test_wca_raining_process():
    new_wca = wca.WCA()

    search_space = search.SearchSpace(n_agents=20, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    flows = new_wca._flow_intensity(search_space.agents)

    new_wca._raining_process(search_space.agents, search_space.best_agent)

    assert search_space.agents[-1].position[0] != 0


def test_wca_update_stream():
    new_wca = wca.WCA()

    search_space = search.SearchSpace(n_agents=20, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    flows = new_wca._flow_intensity(search_space.agents)

    new_wca._update_stream(search_space.agents, search_space.best_agent, flows)

    assert search_space.agents[-1].position[0] != 0


def test_wca_update_river():
    new_wca = wca.WCA()

    search_space = search.SearchSpace(n_agents=20, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    new_wca._update_river(search_space.agents, search_space.best_agent)

    assert search_space.agents[1].position[0] != 0


def test_wca_update():
    new_wca = wca.WCA()

    search_space = search.SearchSpace(n_agents=20, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    flows = new_wca._flow_intensity(search_space.agents)

    new_wca._update(search_space.agents, search_space.best_agent, flows)

    assert search_space.agents[0].position[0] != 0


def test_wca_run():
    def square(x):
        return np.sum(x**2)

    new_function = function.Function(pointer=square)

    new_wca = wca.WCA()

    search_space = search.SearchSpace(n_agents=20, n_iterations=10,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_wca.run(search_space, new_function)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, "The algorithm wca failed to converge"
