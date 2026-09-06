# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

import numpy as np
import pytest

from opytimizer.optimizers.misc import cem
from opytimizer.spaces import search


def test_cem_params():
    params = {
        "n_updates": 5,
        "alpha": 0.7,
    }

    new_cem = cem.CEM(params=params)

    assert new_cem.n_updates == 5

    assert new_cem.alpha == 0.7


@pytest.mark.parametrize("phase", ["construction", "update"])
def test_cem_accepts_numpy_scalar_parameters(phase):
    params = {"n_updates": np.int64(2), "alpha": np.float32(0.5)}
    new_cem = cem.CEM(params if phase == "construction" else None)
    if phase == "update":
        for name, value in params.items():
            setattr(new_cem, name, value)
    search_space = search.SearchSpace(4, 2, [0, 0], [10, 10])
    new_cem.compile(search_space)

    new_cem.update(search_space, lambda x: np.sum(x**2))

    assert new_cem.n_updates is params["n_updates"]
    assert new_cem.alpha is params["alpha"]
    assert np.all(np.isfinite(new_cem.mean))
    assert np.all(np.isfinite(new_cem.std))


def test_cem_compile():
    search_space = search.SearchSpace(n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10])

    new_cem = cem.CEM()
    new_cem.compile(search_space)


def test_cem_create_new_samples():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10])

    new_cem = cem.CEM()
    new_cem.compile(search_space)

    new_cem._create_new_samples(search_space.agents, square)


@pytest.mark.parametrize("alpha", [0, 0.25, 1, 2])
def test_cem_update_mean_preserves_variables(alpha):
    new_cem = cem.CEM({"alpha": alpha})
    new_cem.mean = np.array([-1.0, 99.0])
    updates = np.array([[[0.0, 2.0], [100.0, 102.0]], [[4.0, 6.0], [104.0, 106.0]]])

    result = new_cem._update_mean(updates)

    expected = alpha * np.array([-1.0, 99.0]) + (1 - alpha) * np.array([3.0, 103.0])
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("alpha", [0, 0.25, 1, 2])
def test_cem_update_std_preserves_variables(alpha):
    new_cem = cem.CEM({"alpha": alpha})
    new_cem.mean = np.array([3.0, 103.0])
    new_cem.std = np.array([2.0, 4.0])
    updates = np.array([[[0.0, 2.0], [100.0, 102.0]], [[4.0, 6.0], [104.0, 106.0]]])

    result = new_cem._update_std(updates)

    expected = alpha * np.array([2.0, 4.0]) + (1 - alpha) * np.sqrt(5.0)
    np.testing.assert_allclose(result, expected)


def test_cem_update_uses_updated_mean_for_standard_deviation(monkeypatch):
    search_space = search.SearchSpace(2, 2, [-10, 90], [10, 110])
    for agent, position in zip(search_space.agents, [[0.0, 100.0], [2.0, 102.0]]):
        agent.position[:] = np.array(position)[:, None]
        agent.fit = float(np.sum(agent.position**2))
    new_cem = cem.CEM({"alpha": 0.5, "n_updates": 2})
    new_cem.mean = np.array([-1.0, 99.0])
    new_cem.std = np.array([2.0, 4.0])
    monkeypatch.setattr(new_cem, "_create_new_samples", lambda *args: None)

    new_cem.update(search_space, lambda x: np.sum(x**2))

    np.testing.assert_allclose(new_cem.mean, [0.0, 100.0])
    np.testing.assert_allclose(new_cem.std, [1 + np.sqrt(2) / 2, 2 + np.sqrt(2) / 2])


@pytest.mark.parametrize(
    "name,value,error",
    [
        ("n_updates", 0, ValueError),
        ("n_updates", -1, ValueError),
        ("n_updates", 1.5, TypeError),
        ("n_updates", "2", TypeError),
        ("n_updates", np.nan, TypeError),
        ("alpha", -0.1, ValueError),
        ("alpha", np.nan, ValueError),
        ("alpha", "0.5", TypeError),
    ],
)
@pytest.mark.parametrize("phase", ["construction", "update"])
def test_cem_rejects_invalid_parameters_before_sampling(monkeypatch, name, value, error, phase):
    def unexpected_sampling(*args):
        pytest.fail("invalid parameters reached sampling")

    monkeypatch.setattr(cem.CEM, "_create_new_samples", unexpected_sampling)

    with pytest.raises(error, match=name):
        if phase == "construction":
            cem.CEM({name: value})
        else:
            new_cem = cem.CEM()
            setattr(new_cem, name, value)
            new_cem.update(None, None)


def test_cem_update():
    def square(x):
        return np.sum(x**2)

    new_function = square

    search_space = search.SearchSpace(n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10])

    new_cem = cem.CEM()
    new_cem.compile(search_space)

    new_cem.update(search_space, new_function)
