# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

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
    search_space = search.SearchSpace(n_agents=2, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10])

    new_pso = pso.PSO()
    new_pso.compile(search_space)


def test_pso_evaluate():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=2, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10])

    new_pso = pso.PSO()
    new_pso.compile(search_space)

    new_pso.evaluate(search_space, square)


def test_pso_update():
    search_space = search.SearchSpace(n_agents=2, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10])

    new_pso = pso.PSO()
    new_pso.compile(search_space)

    new_pso.update(search_space)


def test_aiwpso_compute_success():
    search_space = search.SearchSpace(n_agents=2, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10])

    new_aiwpso = pso.AIWPSO()
    new_aiwpso.compile(search_space)

    new_aiwpso.fitness = [1, 1]
    new_aiwpso._compute_success(search_space.agents)


def test_aiwpso_update():
    search_space = search.SearchSpace(n_agents=2, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10])

    new_aiwpso = pso.AIWPSO()
    new_aiwpso.compile(search_space)

    new_aiwpso.update(search_space, 0)


def test_rpso_compile():
    search_space = search.SearchSpace(n_agents=2, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10])

    new_rpso = pso.RPSO()
    new_rpso.compile(search_space)


def test_rpso_update():
    search_space = search.SearchSpace(n_agents=2, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10])

    new_rpso = pso.RPSO()
    new_rpso.compile(search_space)

    new_rpso.update(search_space)


def test_savpso_update():
    search_space = search.SearchSpace(n_agents=2, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10])

    new_savpso = pso.SAVPSO()
    new_savpso.compile(search_space)

    new_savpso.update(search_space)


def test_vpso_compile():
    search_space = search.SearchSpace(n_agents=2, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10])

    new_vpso = pso.VPSO()
    new_vpso.compile(search_space)


def test_vpso_update():
    search_space = search.SearchSpace(n_agents=2, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10])

    new_vpso = pso.VPSO()
    new_vpso.compile(search_space)

    new_vpso.update(search_space)


@pytest.mark.parametrize("optimizer_type", [pso.PSO, pso.AIWPSO, pso.RPSO, pso.SAVPSO, pso.VPSO])
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


@pytest.mark.parametrize("optimizer_type", [pso.PSO, pso.AIWPSO, pso.RPSO, pso.SAVPSO, pso.VPSO])
@pytest.mark.parametrize("name", ["w", "c1", "c2"])
@pytest.mark.parametrize(
    "value,error",
    [("not-numeric", TypeError), (1j, TypeError), (-1, ValueError), (np.nan, ValueError), (np.inf, ValueError)],
)
def test_pso_rejects_invalid_coefficients_at_construction(optimizer_type, name, value, error):
    with pytest.raises(error, match=rf"`{name}`.*\.$"):
        optimizer_type({name: value})


@pytest.mark.parametrize("optimizer_type", [pso.PSO, pso.AIWPSO, pso.RPSO, pso.SAVPSO, pso.VPSO])
@pytest.mark.parametrize("name", ["w", "c1", "c2"])
@pytest.mark.parametrize("boundary", ["compile", "evaluate", "update"])
def test_pso_rejects_reassigned_coefficients_before_side_effects(optimizer_type, name, boundary):
    space = search.SearchSpace(3, 2, [0, 0], [10, 10])
    optimizer = optimizer_type()
    optimizer.compile(space)
    positions = np.array([agent.position.copy() for agent in space.agents])
    velocity = optimizer.velocity
    calls = []
    setattr(optimizer, name, "not-numeric")

    def objective(position):
        calls.append(position)
        return np.sum(position**2)

    with pytest.raises(TypeError, match=rf"`{name}`.*\.$"):
        if boundary == "evaluate":
            optimizer.evaluate(space, objective)
        elif boundary == "update" and isinstance(optimizer, pso.AIWPSO):
            optimizer.update(space, 0)
        else:
            getattr(optimizer, boundary)(space)

    assert calls == []
    assert optimizer.velocity is velocity
    np.testing.assert_array_equal(optimizer.velocity, 0)
    np.testing.assert_array_equal([agent.position for agent in space.agents], positions)


@pytest.mark.parametrize("optimizer_type", [pso.PSO, pso.AIWPSO, pso.RPSO, pso.SAVPSO, pso.VPSO])
@pytest.mark.parametrize("scalar", [float, np.float32, np.float64, np.int64])
def test_pso_accepts_real_scalars_and_inertia_above_one(optimizer_type, scalar):
    optimizer = optimizer_type({"w": scalar(2), "c1": scalar(0), "c2": scalar(1)})
    space = search.SearchSpace(3, 2, [0, 0], [10, 10])
    optimizer.compile(space)
    optimizer.evaluate(space, lambda position: np.sum(position**2))
    if isinstance(optimizer, pso.AIWPSO):
        optimizer.update(space, 0)
    else:
        optimizer.update(space)

    assert np.all(np.isfinite([agent.position for agent in space.agents]))


@pytest.mark.parametrize("name", ["w_min", "w_max"])
@pytest.mark.parametrize(
    "value,error",
    [("not-numeric", TypeError), (1j, TypeError), (-1, ValueError), (np.nan, ValueError), (np.inf, ValueError)],
)
def test_aiwpso_rejects_invalid_limits_at_construction(name, value, error):
    with pytest.raises(error, match=rf"`{name}`.*\.$"):
        pso.AIWPSO({name: value})


def test_aiwpso_rejects_unordered_limits_at_construction():
    with pytest.raises(ValueError, match=r"`w_max`.*`w_min`.*\.$"):
        pso.AIWPSO({"w_min": 2, "w_max": 1})


@pytest.mark.parametrize("boundary", ["compile", "evaluate", "update"])
@pytest.mark.parametrize("value", [-1, np.nan, np.inf, 2])
def test_aiwpso_rejects_reassigned_limits_before_side_effects(boundary, value):
    optimizer = pso.AIWPSO()
    space = search.SearchSpace(3, 2, [0, 0], [10, 10])
    optimizer.compile(space)
    positions = np.array([agent.position.copy() for agent in space.agents])
    optimizer.w_min = value
    calls = []

    def objective(position):
        calls.append(position)
        return np.sum(position**2)

    with pytest.raises(ValueError, match=r"`w_m(?:in|ax)`.*\.$"):
        if boundary == "evaluate":
            optimizer.evaluate(space, objective)
        elif boundary == "update":
            optimizer.update(space, 0)
        else:
            optimizer.compile(space)

    assert calls == []
    assert optimizer.w == 0.7
    np.testing.assert_array_equal([agent.position for agent in space.agents], positions)


def test_aiwpso_update_validates_once(monkeypatch):
    optimizer = pso.AIWPSO()
    space = search.SearchSpace(3, 2, [0, 0], [10, 10])
    optimizer.compile(space)
    validate = optimizer._validate_parameters
    calls = []

    def checked():
        calls.append(None)
        validate()

    monkeypatch.setattr(optimizer, "_validate_parameters", checked)
    optimizer.update(space, 0)

    assert len(calls) == 1


@pytest.mark.parametrize("limit", [np.float32(0), np.float64(2), np.int64(2)])
def test_aiwpso_accepts_equal_nonnegative_real_limits(limit):
    optimizer = pso.AIWPSO({"w_min": limit, "w_max": limit})
    space = search.SearchSpace(3, 2, [0, 0], [10, 10])
    optimizer.compile(space)
    optimizer.update(space, 0)

    assert optimizer.w == limit
