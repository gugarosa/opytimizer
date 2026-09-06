# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

import numpy as np
import pytest

from opytimizer.optimizers.evolutionary import foa
from opytimizer.spaces import search


def test_foa_params():
    params = {
        "life_time": 6,
        "area_limit": 30,
        "LSC": 1,
        "GSC": 1,
        "transfer_rate": 0.1,
    }

    new_foa = foa.FOA(params=params)

    assert new_foa.life_time == 6

    assert new_foa.area_limit == 30

    assert new_foa.LSC == 1

    assert new_foa.GSC == 1

    assert new_foa.transfer_rate == 0.1


def test_foa_compile():
    search_space = search.SearchSpace(n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10])

    new_foa = foa.FOA()
    new_foa.compile(search_space)


def test_foa_local_seeding():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10])

    new_foa = foa.FOA()
    new_foa.compile(search_space)

    new_foa._local_seeding(search_space, square)


def test_foa_population_limiting():
    search_space = search.SearchSpace(n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10])

    new_foa = foa.FOA()
    new_foa.compile(search_space)

    candidate = new_foa._population_limiting(search_space)

    assert len(candidate) == 0

    new_foa.life_time = 1
    new_foa.area_limit = 1
    new_foa.age = [2] * 10
    candidate = new_foa._population_limiting(search_space)

    assert len(candidate) == 9


def test_foa_global_seeding():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10])

    new_foa = foa.FOA()
    new_foa.compile(search_space)

    new_foa.life_time = 1
    new_foa.area_limit = 1
    new_foa.transfer_rate = 0.5
    new_foa.age = [2] * 10
    candidate = new_foa._population_limiting(search_space)

    new_foa._global_seeding(search_space, square, candidate)


def test_foa_update():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10])

    new_foa = foa.FOA()
    new_foa.compile(search_space)

    new_foa.update(search_space, square)


@pytest.mark.parametrize(
    "area_limit, exception",
    [
        (0, ValueError),
        (-1, ValueError),
        (1.5, TypeError),
        (np.float64(2), TypeError),
        ("2", TypeError),
        (None, TypeError),
        (False, ValueError),
    ],
)
def test_foa_rejects_invalid_area_limit(area_limit, exception):
    with pytest.raises(exception, match=r"`area_limit`.*\.$"):
        foa.FOA({"area_limit": area_limit})


@pytest.mark.parametrize("method", ["update", "_population_limiting"])
@pytest.mark.parametrize("area_limit", [0, -1, 1.5])
def test_foa_reassigned_area_limit_fails_before_population_mutation(method, area_limit):
    optimizer = foa.FOA()
    space = search.SearchSpace(n_agents=3, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10])
    optimizer.compile(space)
    optimizer.age = [2, 0, 1]
    for index, agent in enumerate(space.agents):
        agent.fit = 3 - index
    agents = list(space.agents)
    positions = [agent.position.copy() for agent in agents]
    ages = optimizer.age.copy()
    optimizer.area_limit = area_limit

    with pytest.raises((TypeError, ValueError)):
        if method == "update":
            optimizer.update(space, np.sum)
        else:
            optimizer._population_limiting(space)

    assert len(space.agents) == len(agents)
    assert all(actual is expected for actual, expected in zip(space.agents, agents))
    assert optimizer.age == ages
    for agent, position in zip(space.agents, positions):
        np.testing.assert_array_equal(agent.position, position)


@pytest.mark.parametrize("area_limit", [1, np.int64(1), True])
def test_foa_accepts_integral_area_limit(area_limit):
    optimizer = foa.FOA({"area_limit": area_limit})
    space = search.SearchSpace(n_agents=3, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10])
    optimizer.compile(space)

    candidates = optimizer._population_limiting(space)

    assert len(space.agents) == 1
    assert len(optimizer.age) == 1
    assert len(candidates) == 2
