import numpy as np
import pytest

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


@pytest.mark.parametrize(
    "optimizer_type", [pso.PSO, pso.AIWPSO, pso.RPSO, pso.SAVPSO, pso.VPSO]
)
def test_pso_variants_preserve_shared_configuration_and_fresh_state(optimizer_type):
    space = search.SearchSpace(3, 2, [0, 0], [10, 10])
    optimizer = optimizer_type({"w": 0.5, "c1": 1.0, "c2": 2.0})
    optimizer.compile(space)
    previous_velocity = optimizer.velocity
    optimizer.velocity[:] = 3
    optimizer.local_position[:] = 4

    optimizer.compile(space)

    assert (optimizer.w, optimizer.c1, optimizer.c2) == (0.5, 1.0, 2.0)
    assert optimizer.velocity is not previous_velocity
    np.testing.assert_array_equal(optimizer.velocity, np.zeros((3, 2, 1)))
    np.testing.assert_array_equal(optimizer.local_position, np.zeros((3, 2, 1)))
    if isinstance(optimizer, pso.RPSO):
        assert optimizer.mass.shape == (3, 2, 1)
        assert np.all((optimizer.mass >= 0) & (optimizer.mass < 1))
    if isinstance(optimizer, pso.VPSO):
        np.testing.assert_array_equal(optimizer.v_velocity, np.ones((3, 2, 1)))
