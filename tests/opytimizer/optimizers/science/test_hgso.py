import numpy as np
import pytest

from opytimizer import Opytimizer
from opytimizer.core import Space
from opytimizer.optimizers.science import hgso
from opytimizer.spaces import search

np.random.seed(0)


def test_hgso_params():
    params = {
        "n_clusters": 2,
        "l1": 0.0005,
        "l2": 100,
        "l3": 0.001,
        "alpha": 1.0,
        "beta": 1.0,
        "K": 1.0,
    }

    new_hgso = hgso.HGSO(params=params)

    assert new_hgso.n_clusters == 2

    assert new_hgso.l1 == 0.0005

    assert new_hgso.l2 == 100

    assert new_hgso.l3 == 0.001

    assert new_hgso.alpha == 1.0

    assert new_hgso.beta == 1.0

    assert new_hgso.K == 1.0


def test_hgso_compile():
    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_hgso = hgso.HGSO()
    new_hgso.compile(search_space)


def test_hgso_update_position():
    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_hgso = hgso.HGSO()
    new_hgso.compile(search_space)

    position = new_hgso._update_position(
        search_space.agents[0], search_space.agents[1], search_space.best_agent, 0.5
    )

    assert position[0][0] != 0


def test_hgso_update():
    def square(x):
        return np.sum(x**2)

    search_space = search.SearchSpace(
        n_agents=10, n_variables=2, lower_bound=[0, 0], upper_bound=[10, 10]
    )

    new_hgso = hgso.HGSO()
    new_hgso.compile(search_space)

    new_hgso.update(search_space, square, 1, 10)


def make_space(n_agents, n_dimensions):
    space = Space(n_agents, 2, n_dimensions, [0, 10], [1, 20])
    space.build()
    for i, agent in enumerate(space.agents):
        fraction = (i + 1) / (n_agents + 1)
        agent.position[:] = [[fraction], [10 + 10 * fraction]]
    return space


@pytest.mark.parametrize(
    "n_agents,n_clusters,n_dimensions",
    [(1, 1, 1), (5, 2, 1), (7, 3, 1), (3, 3, 2), (10, 2, 1), (10, np.int64(2), 3)],
)
def test_hgso_run_preserves_shapes_bounds_and_fitness(
    n_agents, n_clusters, n_dimensions
):
    space = make_space(n_agents, n_dimensions)

    def objective(position):
        assert position.shape == (2, n_dimensions)
        return float(np.sum(position**2))

    model = Opytimizer(space, hgso.HGSO({"n_clusters": n_clusters}), objective)
    model.start(2)

    for agent in [*space.agents, space.best_agent]:
        assert agent.position.shape == (2, n_dimensions)
        assert np.all(agent.position >= space.lb[:, None])
        assert np.all(agent.position <= space.ub[:, None])
        assert agent.fit == objective(agent.position)


def test_hgso_cluster_count_takes_effect_when_recompiled():
    space = make_space(6, 1)
    optimizer = hgso.HGSO()
    model = Opytimizer(space, optimizer, lambda x: float(np.sum(x**2)))
    optimizer.n_clusters = 3

    model.start()
    assert optimizer.pressure.shape == (2, 3)

    optimizer.compile(space)
    model.start()
    assert optimizer.pressure.shape == (3, 2)


@pytest.mark.parametrize(
    "n_agents,n_dimensions,n_replacements", [(4, 1, 0), (10, 1, 1), (10, 3, 1)]
)
def test_hgso_replaces_only_worst_agents_with_matching_fitness(
    monkeypatch, n_agents, n_dimensions, n_replacements
):
    space = make_space(n_agents, n_dimensions)
    optimizer = hgso.HGSO()
    optimizer.compile(space)
    optimizer.evaluate(space, lambda x: float(np.sum(x**2)))
    original = {id(agent): agent.position.copy() for agent in space.agents}
    worst = {id(space.agents[-1])} if n_replacements else set()
    evaluated = []

    def objective(position):
        evaluated.append(position.copy())
        return float(np.sum(position**2))

    monkeypatch.setattr(
        optimizer, "_update_position", lambda agent, *args: agent.position.copy()
    )
    monkeypatch.setattr(
        np.random,
        "uniform",
        lambda low=0.0, high=1.0, size=None: (
            np.full(size, 0.5) if size is not None else 0.5
        ),
    )

    optimizer.update(space, objective, 0, 2)

    assert len(evaluated) == n_agents + n_replacements
    for agent in space.agents:
        expected = (
            np.broadcast_to([[0.5], [15]], (2, n_dimensions))
            if id(agent) in worst
            else original[id(agent)]
        )
        np.testing.assert_array_equal(agent.position, expected)
        assert agent.fit == float(np.sum(agent.position**2))


@pytest.mark.parametrize(
    "n_clusters,error",
    [(0, ValueError), (-1, ValueError), (2.0, TypeError), (6, ValueError)],
)
def test_hgso_rejects_invalid_cluster_counts_before_allocation(n_clusters, error):
    optimizer = hgso.HGSO({"n_clusters": n_clusters})

    with pytest.raises(error, match="`n_clusters`"):
        optimizer.compile(make_space(5, 1))
