import numpy as np

from opytimizer.optimizers.swarm import pso
from opytimizer.spaces import search


def test_pso_params():
    params = {
        'w': 2,
        'c1': 1.7,
        'c2': 1.7
    }

    new_pso = pso.PSO(params=params)

    assert new_pso.w == 2

    assert new_pso.c1 == 1.7

    assert new_pso.c2 == 1.7


def test_pso_params_setter():
    new_pso = pso.PSO()

    try:
        new_pso.w = 'a'
    except:
        new_pso.w = 1

    try:
        new_pso.w = -1
    except:
        new_pso.w = 1

    assert new_pso.w == 1

    try:
        new_pso.c1 = 'b'
    except:
        new_pso.c1 = 1.5

    try:
        new_pso.c1 = -1
    except:
        new_pso.c1 = 1.5

    assert new_pso.c1 == 1.5

    try:
        new_pso.c2 = 'c'
    except:
        new_pso.c2 = 1.5

    try:
        new_pso.c2 = -1
    except:
        new_pso.c2 = 1.5

    assert new_pso.c2 == 1.5


def test_pso_create_additional_attrs():
    search_space = search.SearchSpace(n_agents=2, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_pso = pso.PSO()
    new_pso.create_additional_attrs(search_space)

    try:
        new_pso.local_position = 1
    except:
        new_pso.local_position = np.array([1])

    assert new_pso.local_position == np.array([1])

    try:
        new_pso.velocity = 1
    except:
        new_pso.velocity = np.array([1])

    assert new_pso.velocity == np.array([1])


def test_pso_evaluate():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=2, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_pso = pso.PSO()
    new_pso.create_additional_attrs(search_space)

    new_pso.evaluate(search_space, square)


def test_pso_update():
    search_space = search.SearchSpace(n_agents=2, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_pso = pso.PSO()
    new_pso.create_additional_attrs(search_space)

    new_pso.update(search_space)


def test_aiwpso_compute_success():
    search_space = search.SearchSpace(n_agents=2, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_aiwpso = pso.AIWPSO()
    new_aiwpso.create_additional_attrs(search_space)

    new_aiwpso.fitness = [1, 1]
    new_aiwpso._compute_success(search_space.agents)


def test_aiwpso_update():
    search_space = search.SearchSpace(n_agents=2, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_aiwpso = pso.AIWPSO()
    new_aiwpso.create_additional_attrs(search_space)

    new_aiwpso.update(search_space, 0)


def test_rpso_create_additional_attrs():
    search_space = search.SearchSpace(n_agents=2, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_rpso = pso.RPSO()
    new_rpso.create_additional_attrs(search_space)

    try:
        new_rpso.local_position = 1
    except:
        new_rpso.local_position = np.array([1])

    assert new_rpso.local_position == np.array([1])

    try:
        new_rpso.velocity = 1
    except:
        new_rpso.velocity = np.array([1])

    assert new_rpso.velocity == np.array([1])

    try:
        new_rpso.mass = 1
    except:
        new_rpso.mass = np.array([1])

    assert new_rpso.mass == np.array([1])


def test_rpso_update():
    search_space = search.SearchSpace(n_agents=2, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_rpso = pso.RPSO()
    new_rpso.create_additional_attrs(search_space)

    new_rpso.update(search_space)


def test_savpso_update():
    search_space = search.SearchSpace(n_agents=2, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_savpso = pso.SAVPSO()
    new_savpso.create_additional_attrs(search_space)

    new_savpso.update(search_space)


def test_vpso_create_additional_attrs():
    search_space = search.SearchSpace(n_agents=2, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_vpso = pso.VPSO()
    new_vpso.create_additional_attrs(search_space)

    try:
        new_vpso.local_position = 1
    except:
        new_vpso.local_position = np.array([1])

    assert new_vpso.local_position == np.array([1])

    try:
        new_vpso.velocity = 1
    except:
        new_vpso.velocity = np.array([1])

    assert new_vpso.velocity == np.array([1])

    try:
        new_vpso.v_velocity = 1
    except:
        new_vpso.v_velocity = np.array([1])

    assert new_vpso.v_velocity == np.array([1])


def test_vpso_update():
    search_space = search.SearchSpace(n_agents=2, n_variables=2,
                                      lower_bound=[0, 0], upper_bound=[10, 10])

    new_vpso = pso.VPSO()
    new_vpso.create_additional_attrs(search_space)

    new_vpso.update(search_space)
