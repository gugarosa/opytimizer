import sys
from types import MappingProxyType

import numpy as np
import pytest

from opytimizer.core import Optimizer
from opytimizer.spaces import SearchSpace


def test_optimizer_build_applies_mapping_without_lifecycle_state():
    optimizer = Optimizer()

    optimizer.build({"rate": 0.5})

    assert optimizer.rate == 0.5
    assert not hasattr(optimizer, "algorithm")
    assert not hasattr(optimizer, "params")
    assert not hasattr(optimizer, "built")


@pytest.mark.parametrize("params", [[], (), "", 0, False, ["rate"]])
def test_optimizer_build_rejects_non_mappings(params):
    optimizer = Optimizer()

    with pytest.raises(TypeError, match="params"):
        optimizer.build(params)

    assert vars(optimizer) == {}


def test_optimizer_build_preserves_mapping_semantics():
    class FalseyMapping(dict):
        def __bool__(self):
            return False

    optimizer = Optimizer()
    optimizer.build(None)
    optimizer.build({})
    assert vars(optimizer) == {}

    optimizer.build(MappingProxyType({"rate": 0.5}))
    optimizer.build(FalseyMapping(rate=0.25))
    assert optimizer.rate == 0.25


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
