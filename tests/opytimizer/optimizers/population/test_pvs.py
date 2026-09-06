# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

import numpy as np
import pytest

from opytimizer.math import random
from opytimizer.optimizers.population import pvs
from opytimizer.spaces import search


def test_pvs_update():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10])

    new_pvs = pvs.PVS()

    new_pvs.update(search_space, square)


@pytest.mark.parametrize("n_agents", [1, 2])
def test_pvs_rejects_insufficient_peers_before_sorting_or_sampling(monkeypatch, n_agents):
    search_space = search.SearchSpace(n_agents, 1, [0], [10])
    for i, agent in enumerate(search_space.agents):
        agent.fit = float(n_agents - i)
    original_agents = list(search_space.agents)

    def unexpected_sampling(*args, **kwargs):
        pytest.fail("insufficient population reached sampling")

    monkeypatch.setattr(random, "integer", unexpected_sampling)
    monkeypatch.setattr(np.random, "choice", unexpected_sampling)

    with pytest.raises(ValueError, match="at least 3"):
        pvs.PVS().update(search_space, lambda x: np.sum(x**2))

    assert search_space.agents == original_agents


def test_pvs_samples_two_peers_without_replacement(monkeypatch):
    search_space = search.SearchSpace(4, 1, [0], [10])
    for i, agent in enumerate(search_space.agents):
        agent.position[:] = i + 1
        agent.fit = float((i + 1) ** 2)
    calls = []
    peer_indexes = []

    class TrackedAgents(list):
        def __getitem__(self, index):
            peer_indexes[-1].append(index)
            return super().__getitem__(index)

    def choice(n, size, replace):
        calls.append((n, size, replace))
        peer_indexes.append([])
        return np.array([0, 1])

    def unexpected_rejection_sampling(*args, **kwargs):
        pytest.fail("peer selection used rejection sampling")

    monkeypatch.setattr(np.random, "choice", choice)
    monkeypatch.setattr(np.random, "uniform", lambda *args: np.array([0.25]))
    monkeypatch.setattr(random, "integer", unexpected_rejection_sampling)
    search_space.agents = TrackedAgents(search_space.agents)

    pvs.PVS().update(search_space, lambda x: 1000.0)

    assert calls == [(3, 2, False)] * 4
    assert [indexes[:2] for indexes in peer_indexes] == [[1, 2], [0, 2], [0, 1], [0, 1]]
    for i, indexes in enumerate(peer_indexes):
        assert i not in indexes
