import numpy as np

from opytimizer.optimizers.population import aeo
from opytimizer.spaces import search


def test_aeo_production():
    new_aeo = aeo.AEO()

    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    a = new_aeo._production(
        search_space.agents[0], search_space.best_agent, 1, 10)

    assert type(a).__name__ == 'Agent'


def test_aeo_herbivore_consumption():
    new_aeo = aeo.AEO()

    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    a = new_aeo._herbivore_consumption(
        search_space.agents[0], search_space.agents[1], 0.5)

    assert type(a).__name__ == 'Agent'


def test_aeo_omnivore_consumption():
    new_aeo = aeo.AEO()

    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    a = new_aeo._omnivore_consumption(
        search_space.agents[0], search_space.agents[1], search_space.agents[2], 0.5)

    assert type(a).__name__ == 'Agent'


def test_aeo_carnivore_consumption():
    new_aeo = aeo.AEO()

    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    a = new_aeo._carnivore_consumption(
        search_space.agents[0], search_space.agents[1], 0.5)

    assert type(a).__name__ == 'Agent'


def test_aeo_update_composition():
    def square(x):
        return np.sum(x**2)

    new_aeo = aeo.AEO()

    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_aeo._update_composition(
        search_space.agents, search_space.best_agent, square, 1, 10)


def test_aeo_update_decomposition():
    def square(x):
        return np.sum(x**2)

    new_aeo = aeo.AEO()

    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_aeo._update_decomposition(
        search_space.agents, search_space.best_agent, square)


def test_aeo_update():
    def square(x):
        return np.sum(x**2)

    new_aeo = aeo.AEO()

    search_space = search.SearchSpace(n_agents=10, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_aeo.update(search_space, square, 1, 10)
