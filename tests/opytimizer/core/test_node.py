import numpy as np
import pytest

from opytimizer.core import Node
from opytimizer.core.node import _evaluate


def terminal(name, value=1):
    return Node(name, "TERMINAL", np.array(value))


def binary_tree(name):
    left = terminal(1)
    right = terminal(2)
    root = Node(name, "FUNCTION", left=left, right=right)
    left.parent = root
    right.parent = root
    right.flag = False
    return root


def test_node_uses_compact_representation():
    node = Node("SUM", "FUNCTION")

    assert repr(node) == "FUNCTION:SUM:True"
    assert str(node) == repr(node)


@pytest.mark.parametrize(
    "args,error",
    [
        ((0.0, "FUNCTION"), TypeError),
        ((0, "UNKNOWN"), ValueError),
        ((0, "TERMINAL", 1), TypeError),
        ((0, "FUNCTION", None, 1), TypeError),
        ((0, "FUNCTION", None, None, 1), TypeError),
        ((0, "FUNCTION", None, None, None, 1), TypeError),
    ],
)
def test_node_validates_constructor_inputs(args, error):
    with pytest.raises(error):
        Node(*args)


def test_node_traversal_properties_and_lookup():
    root = binary_tree("SUM")

    assert root.pre_order == [root, root.left, root.right]
    assert root.post_order == [root.left, root.right, root]
    assert root.min_depth == 1
    assert root.max_depth == 1
    assert root.n_leaves == 2
    assert root.n_nodes == 3
    assert root.find_node(1) == (root, True)
    assert root.find_node(2) == (root, False)
    assert root.find_node(3) == (None, False)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("SUM", 2),
        ("SUB", 0),
        ("MUL", 1),
        ("DIV", 1),
        ("EXP", np.e),
        ("SQRT", 1),
        ("LOG", 0),
        ("ABS", 1),
        ("SIN", np.sin(1)),
        ("COS", np.cos(1)),
    ],
)
def test_node_evaluation(name, expected):
    node = binary_tree(name)

    assert np.allclose(_evaluate(node), expected)
    assert np.allclose(node.position, expected)
