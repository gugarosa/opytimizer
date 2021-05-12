import numpy as np

from opytimizer.optimizers.swarm import sos
from opytimizer.spaces import search


def test_sos_mutualism():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_sos = sos.SOS()

    new_sos._mutualism(
        search_space.agents[0], search_space.agents[1], search_space.best_agent, square)


def test_sos_commensalism():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_sos = sos.SOS()

    new_sos._commensalism(
        search_space.agents[0], search_space.agents[1], search_space.best_agent, square)


def test_sos_parasitism():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_sos = sos.SOS()

    new_sos._parasitism(search_space.agents[0], search_space.agents[1], square)


def test_sos_update():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_sos = sos.SOS()

    new_sos.update(search_space, square)
