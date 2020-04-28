import numpy as np

from opytimizer.core import function
from opytimizer.optimizers.science import gsa
from opytimizer.spaces import search
from opytimizer.utils import constants


def test_gsa_hyperparams():
    hyperparams = {
        'G': 2.467,
    }

    new_gsa = gsa.GSA(hyperparams=hyperparams)

    assert new_gsa.G == 2.467


def test_gsa_hyperparams_setter():
    new_gsa = gsa.GSA()

    try:
        new_gsa.G = 'a'
    except:
        new_gsa.G = 0.1

    try:
        new_gsa.G = -1
    except:
        new_gsa.G = 0.1

    assert new_gsa.G == 0.1


def test_gsa_build():
    new_gsa = gsa.GSA()

    assert new_gsa.built == True


def test_gsa_calculate_mass():
    new_gsa = gsa.GSA()

    search_space = search.SearchSpace(n_agents=10, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    search_space.agents[0].fit = 1

    search_space.agents.sort(key=lambda x: x.fit)

    mass = new_gsa._calculate_mass(search_space.agents)

    assert len(mass) > 0


def test_gsa_calculate_force():
    new_gsa = gsa.GSA()

    search_space = search.SearchSpace(n_agents=10, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    search_space.agents[0].fit = 1

    search_space.agents.sort(key=lambda x: x.fit)

    mass = new_gsa._calculate_mass(search_space.agents)

    gravity = 1

    force = new_gsa._calculate_force(search_space.agents, mass, gravity)

    assert force.shape[0] > 0


def test_gsa_update_velocity():
    new_gsa = gsa.GSA()

    search_space = search.SearchSpace(n_agents=10, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    search_space.agents[0].fit = 1

    search_space.agents.sort(key=lambda x: x.fit)

    mass = new_gsa._calculate_mass(search_space.agents)

    gravity = 1

    force = new_gsa._calculate_force(search_space.agents, mass, gravity)

    velocity = new_gsa._update_velocity(force[0], mass[0], 1)

    assert velocity[0] != 0


def test_gsa_update_position():
    new_gsa = gsa.GSA()

    position = new_gsa._update_position(1, 1)

    assert position == 2


def test_gsa_run():
    def square(x):
        return np.sum(x**2)

    def hook(optimizer, space, function):
        return

    new_function = function.Function(pointer=square)

    hyperparams = {
        'G': 100
    }

    new_gsa = gsa.GSA(hyperparams=hyperparams)

    search_space = search.SearchSpace(n_agents=10, n_iterations=100,
                                      n_variables=2, lower_bound=[0, 0],
                                      upper_bound=[10, 10])

    history = new_gsa.run(search_space, new_function, pre_evaluation_hook=hook)

    assert len(history.agents) > 0
    assert len(history.best_agent) > 0

    best_fitness = history.best_agent[-1][1]
    assert best_fitness <= constants.TEST_EPSILON, 'The algorithm gsa failed to converge.'
