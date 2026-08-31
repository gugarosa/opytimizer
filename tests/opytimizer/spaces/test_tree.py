import numpy as np
import pytest

from opytimizer.core import Node
from opytimizer.spaces import TreeSpace


def test_tree_space_builds_trees_agents_and_terminals():
    np.random.seed(0)
    space = TreeSpace(
        2, 1, 0, 1, n_terminals=2, min_depth=1, max_depth=3, functions=["SUM"]
    )

    assert len(space.terminals) == 2
    assert len(space.trees) == 2
    assert len(space.agents) == 2
    assert isinstance(space.best_tree, Node)
    assert not hasattr(space, "built")


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"n_terminals": 1.0}, TypeError),
        ({"n_terminals": 0}, ValueError),
        ({"min_depth": 1.0}, TypeError),
        ({"min_depth": 0}, ValueError),
        ({"max_depth": 1.0}, TypeError),
        ({"min_depth": 2, "max_depth": 1}, ValueError),
        ({"functions": "SUM"}, TypeError),
        ({"functions": ["UNKNOWN"]}, ValueError),
    ],
)
def test_tree_space_validates_constructor_inputs(kwargs, error):
    with pytest.raises(error):
        TreeSpace(1, 1, 0, 1, **kwargs)


def test_tree_space_grow_returns_terminal_at_max_depth():
    space = TreeSpace(1, 1, 0, 1, max_depth=1)

    tree = space.grow(1, 1)

    assert isinstance(tree, Node)
    assert tree.category == "TERMINAL"
