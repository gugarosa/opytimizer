import sys

import numpy as np

from opytimizer.core import Optimizer
from opytimizer.spaces import SearchSpace


def test_optimizer_build_applies_mapping_without_lifecycle_state():
    optimizer = Optimizer()

    optimizer.build({"rate": 0.5})

    assert optimizer.rate == 0.5
    assert not hasattr(optimizer, "algorithm")
    assert not hasattr(optimizer, "params")
    assert not hasattr(optimizer, "built")


def test_optimizer_base_hooks_are_noops():
    optimizer = Optimizer()

    assert optimizer.compile(None) is None
    assert optimizer.update() is None


def test_optimizer_evaluates_raw_callable():
    space = SearchSpace(2, 2, [0, 0], [1, 1])
    space.agents[0].position[:] = 0.5
    space.agents[1].position[:] = 1

    Optimizer().evaluate(space, lambda x: np.sum(x**2))

    assert space.best_agent.fit == 0.5
    assert space.best_agent.fit < sys.float_info.max
    assert np.array_equal(space.best_agent.position, space.agents[0].position)
