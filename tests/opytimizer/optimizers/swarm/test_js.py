# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

import numpy as np
import pytest

from opytimizer.optimizers.swarm import js
from opytimizer.spaces import search

np.random.seed(0)


def test_js_params():
    params = {"eta": 4.0, "beta": 3.0, "gamma": 0.1}

    new_js = js.JS(params=params)

    assert new_js.eta == 4.0

    assert new_js.beta == 3.0

    assert new_js.gamma == 0.1


def test_js_initialize_chaotic_map():
    search_space = search.SearchSpace(n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10])

    new_js = js.JS()
    new_js._initialize_chaotic_map(search_space.agents)


def test_js_compile():
    search_space = search.SearchSpace(n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10])

    new_js = js.JS()
    new_js.compile(search_space)


def test_js_ocean_current():
    search_space = search.SearchSpace(n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10])

    new_js = js.JS()
    new_js.compile(search_space)

    trend = new_js._ocean_current(search_space.agents, search_space.best_agent)

    assert trend[0][0] != 0


def test_js_motion_a():
    search_space = search.SearchSpace(n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10])

    new_js = js.JS()
    new_js.compile(search_space)

    motion = new_js._motion_a(0, 1)

    assert motion[0] != 0


def test_js_motion_b():
    search_space = search.SearchSpace(n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10])

    new_js = js.JS()
    new_js.compile(search_space)

    motion = new_js._motion_b(search_space.agents[0], search_space.agents[1])

    assert motion[0][0] != 0


def test_js_update():
    search_space = search.SearchSpace(n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10])

    new_js = js.JS()
    new_js.compile(search_space)

    new_js.update(search_space, 1, 10)


def test_nbjs_motion_a():
    search_space = search.SearchSpace(n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10])

    new_nbjs = js.NBJS()
    new_nbjs.compile(search_space)

    motion = new_nbjs._motion_a(0, 1)

    assert motion[0] != 0


@pytest.mark.parametrize("optimizer_type", [js.JS, js.NBJS])
@pytest.mark.parametrize("name", ["eta", "beta", "gamma"])
@pytest.mark.parametrize(
    "value,error",
    [
        ("not-numeric", TypeError),
        (1j, TypeError),
        (-4, ValueError),
        (0, ValueError),
        (np.nan, ValueError),
        (np.inf, ValueError),
    ],
)
def test_js_rejects_invalid_coefficients_at_construction(optimizer_type, name, value, error):
    with pytest.raises(error, match=rf"`{name}`.*\.$"):
        optimizer_type({name: value})


@pytest.mark.parametrize("optimizer_type", [js.JS, js.NBJS])
def test_js_rejects_logistic_coefficient_above_four(optimizer_type):
    with pytest.raises(ValueError, match=r"`eta`.*\.$"):
        optimizer_type({"eta": np.nextafter(4.0, np.inf)})


@pytest.mark.parametrize("optimizer_type", [js.JS, js.NBJS])
@pytest.mark.parametrize("boundary", ["compile", "update"])
@pytest.mark.parametrize("name,value", [("eta", -4), ("eta", 5), ("beta", -1), ("gamma", -1)])
def test_js_rejects_reassigned_coefficients_before_mutation(optimizer_type, boundary, name, value):
    optimizer = optimizer_type()
    space = search.SearchSpace(20, 2, [0, 0], [10, 10])
    optimizer.compile(space)
    positions = np.array([agent.position.copy() for agent in space.agents])
    setattr(optimizer, name, value)

    with pytest.raises(ValueError, match=rf"`{name}`.*\.$"):
        if boundary == "update":
            optimizer.update(space, 1, 10)
        else:
            optimizer.compile(space)

    np.testing.assert_array_equal([agent.position for agent in space.agents], positions)


@pytest.mark.parametrize("optimizer_type", [js.JS, js.NBJS])
@pytest.mark.parametrize("draw", [0.75, 0.9])
def test_js_public_operations_validate_once(optimizer_type, draw, monkeypatch):
    optimizer = optimizer_type()
    space = search.SearchSpace(5, 2, [0, 0], [10, 10])
    validate = optimizer._validate_parameters
    calls = []

    def checked():
        calls.append(None)
        validate()

    monkeypatch.setattr(optimizer, "_validate_parameters", checked)
    optimizer.compile(space)

    assert len(calls) == 1

    calls.clear()
    monkeypatch.setattr(np.random, "uniform", lambda low, high, size: np.full(size, draw))
    optimizer.update(space, 1, 10)

    assert len(calls) == 1


@pytest.mark.parametrize("optimizer_type", [js.JS, js.NBJS])
@pytest.mark.parametrize("eta", [np.float32(0.1), np.float64(4), np.int64(4)])
def test_js_logistic_map_preserves_unit_interval_for_supported_coefficients(optimizer_type, eta, monkeypatch):
    optimizer = optimizer_type({"eta": eta, "beta": np.float32(3), "gamma": np.float64(0.1)})
    space = search.SearchSpace(100, 1, [0], [1])
    monkeypatch.setattr(np.random, "uniform", lambda low, high, size: np.full(size, 0.5))

    optimizer.compile(space)

    positions = np.array([agent.position for agent in space.agents])
    assert positions[0, 0, 0] == 0.5
    assert positions[1, 0, 0] == eta * 0.25
    assert np.all(np.isfinite(positions))
    assert np.all((positions >= 0) & (positions <= 1))


@pytest.mark.parametrize("optimizer_type", [js.JS, js.NBJS])
def test_js_motion_coefficients_have_no_logistic_upper_bound(optimizer_type):
    optimizer = optimizer_type({"beta": np.float32(10), "gamma": np.int64(10)})
    space = search.SearchSpace(3, 2, [0, 0], [10, 10])
    optimizer.compile(space)
    optimizer.update(space, 1, 10)

    assert optimizer.beta == 10
    assert optimizer.gamma == 10
    assert np.all(np.isfinite([agent.position for agent in space.agents]))
