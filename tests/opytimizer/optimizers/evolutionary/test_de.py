# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

import numpy as np
import pytest

from opytimizer.optimizers.evolutionary import de
from opytimizer.spaces import search


def test_de_params():
    params = {"CR": 0.9, "F": 0.7}

    new_de = de.DE(params=params)

    assert new_de.CR == 0.9

    assert new_de.F == 0.7


@pytest.mark.parametrize("phase", ["construction", "update"])
def test_de_accepts_numpy_scalar_parameters(phase):
    params = {"CR": np.float32(0.5), "F": np.float32(0.75)}
    new_de = de.DE(params if phase == "construction" else None)
    if phase == "update":
        for name, value in params.items():
            setattr(new_de, name, value)
    search_space = search.SearchSpace(4, 2, [0, 0], [10, 10])

    new_de.update(search_space, lambda x: np.sum(x**2))

    assert new_de.CR is params["CR"]
    assert new_de.F is params["F"]
    assert all(np.isfinite(agent.fit) for agent in search_space.agents)


@pytest.mark.parametrize(
    "name,value,error",
    [
        ("CR", -0.1, ValueError),
        ("CR", 1.1, ValueError),
        ("CR", np.nan, ValueError),
        ("CR", np.inf, ValueError),
        ("CR", "0.5", TypeError),
        ("F", -0.1, ValueError),
        ("F", 2.1, ValueError),
        ("F", np.nan, ValueError),
        ("F", np.inf, ValueError),
        ("F", "0.5", TypeError),
    ],
)
@pytest.mark.parametrize("phase", ["construction", "update"])
def test_de_rejects_invalid_parameters_before_sampling(monkeypatch, name, value, error, phase):
    def unexpected_sampling(*args, **kwargs):
        pytest.fail("invalid parameters reached sampling")

    monkeypatch.setattr(np.random, "choice", unexpected_sampling)

    with pytest.raises(error, match=name):
        if phase == "construction":
            de.DE({name: value})
        else:
            new_de = de.DE()
            setattr(new_de, name, value)
            new_de.update(search.SearchSpace(4, 1, [0], [10]), None)


@pytest.mark.parametrize("CR,F", [(0, 0), (1, 2)])
def test_de_accepts_parameter_boundaries(CR, F):
    new_de = de.DE({"CR": CR, "F": F})

    assert new_de.CR == CR
    assert new_de.F == F


def test_de_mutate_agent():
    new_de = de.DE()

    search_space = search.SearchSpace(n_agents=4, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10])

    agent = new_de._mutate_agent(
        search_space.agents[0],
        search_space.agents[1],
        search_space.agents[2],
        search_space.agents[3],
    )

    assert agent.position[0][0] != 0


def test_de_update():
    def square(x):
        return np.sum(x**2)

    new_de = de.DE()

    search_space = search.SearchSpace(n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10])

    new_de.update(search_space, square)
