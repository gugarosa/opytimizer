import numpy as np

from opytimizer.optimizers.swarm import pso
from opytimizer.spaces import search


def test_pso_params():
    params = {"w": 2, "c1": 1.7, "c2": 1.7}

    new_pso = pso.PSO(params=params)

    assert new_pso.w == 2

    assert new_pso.c1 == 1.7

    assert new_pso.c2 == 1.7


def test_pso_compile():
    search_space = search.SearchSpace(
        n_agents=2, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_pso = pso.PSO()
    new_pso.compile(search_space)


def test_pso_evaluate():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(
        n_agents=2, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_pso = pso.PSO()
    new_pso.compile(search_space)

    new_pso.evaluate(search_space, square)


def test_pso_update():
    search_space = search.SearchSpace(
        n_agents=2, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_pso = pso.PSO()
    new_pso.compile(search_space)

    new_pso.update(search_space)


def test_aiwpso_compute_success():
    search_space = search.SearchSpace(
        n_agents=2, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_aiwpso = pso.AIWPSO()
    new_aiwpso.compile(search_space)

    new_aiwpso.fitness = [1, 1]
    new_aiwpso._compute_success(search_space.agents)


def test_aiwpso_update():
    search_space = search.SearchSpace(
        n_agents=2, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_aiwpso = pso.AIWPSO()
    new_aiwpso.compile(search_space)

    new_aiwpso.update(search_space, 0)


def test_rpso_compile():
    search_space = search.SearchSpace(
        n_agents=2, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_rpso = pso.RPSO()
    new_rpso.compile(search_space)


def test_rpso_update():
    search_space = search.SearchSpace(
        n_agents=2, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_rpso = pso.RPSO()
    new_rpso.compile(search_space)

    new_rpso.update(search_space)


def test_savpso_update():
    search_space = search.SearchSpace(
        n_agents=2, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_savpso = pso.SAVPSO()
    new_savpso.compile(search_space)

    new_savpso.update(search_space)


def test_vpso_compile():
    search_space = search.SearchSpace(
        n_agents=2, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_vpso = pso.VPSO()
    new_vpso.compile(search_space)


def test_vpso_update():
    search_space = search.SearchSpace(
        n_agents=2, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_vpso = pso.VPSO()
    new_vpso.compile(search_space)

    new_vpso.update(search_space)
