import pytest

from opytimizer.core import node
from opytimizer.spaces import tree


def test_tree_space_n_trees():
    new_tree_space = tree.TreeSpace(n_trees=1)

    assert new_tree_space.n_trees == 1


def test_tree_space_n_trees_setter():
    try:
        new_tree_space = tree.TreeSpace(n_trees=0.0)
    except:
        new_tree_space = tree.TreeSpace(n_trees=1)

    try:
        new_tree_space = tree.TreeSpace(n_trees=0)
    except:
        new_tree_space = tree.TreeSpace(n_trees=1)

    assert new_tree_space.n_trees == 1


def test_tree_space_n_terminals():
    new_tree_space = tree.TreeSpace(n_terminals=1)

    assert new_tree_space.n_terminals == 1


def test_tree_space_n_terminals_setter():
    try:
        new_tree_space = tree.TreeSpace(n_terminals=0.0)
    except:
        new_tree_space = tree.TreeSpace(n_terminals=1)

    try:
        new_tree_space = tree.TreeSpace(n_terminals=0)
    except:
        new_tree_space = tree.TreeSpace(n_terminals=1)

    assert new_tree_space.n_terminals == 1


def test_tree_space_min_depth():
    new_tree_space = tree.TreeSpace(min_depth=1)

    assert new_tree_space.min_depth == 1


def test_tree_space_min_depth_setter():
    try:
        new_tree_space = tree.TreeSpace(min_depth=0.0)
    except:
        new_tree_space = tree.TreeSpace(min_depth=1)

    try:
        new_tree_space = tree.TreeSpace(min_depth=0)
    except:
        new_tree_space = tree.TreeSpace(min_depth=1)

    assert new_tree_space.min_depth == 1


def test_tree_space_max_depth():
    new_tree_space = tree.TreeSpace(max_depth=1)

    assert new_tree_space.max_depth == 1


def test_tree_space_max_depth_setter():
    try:
        new_tree_space = tree.TreeSpace(max_depth=0.0)
    except:
        new_tree_space = tree.TreeSpace(max_depth=1)

    try:
        new_tree_space = tree.TreeSpace(max_depth=0)
    except:
        new_tree_space = tree.TreeSpace(max_depth=1)

    assert new_tree_space.max_depth == 1


def test_tree_space_functions():
    new_tree_space = tree.TreeSpace(functions=['SUM'])

    assert len(new_tree_space.functions) == 1


def test_tree_space_functions_setter():
    try:
        new_tree_space = tree.TreeSpace(functions='a')
    except:
        new_tree_space = tree.TreeSpace(functions=['SUM'])

    assert len(new_tree_space.functions) == 1


def test_tree_space_terminals():
    new_tree_space = tree.TreeSpace()

    assert len(new_tree_space.terminals) == 1


def test_tree_space_terminals_setter():
    try:
        new_tree_space = tree.TreeSpace()
        new_tree_space.terminals = 'a'
    except:
        new_tree_space = tree.TreeSpace()
        new_tree_space.terminals = []

    assert len(new_tree_space.terminals) == 0


def test_tree_space_trees():
    new_tree_space = tree.TreeSpace()

    assert len(new_tree_space.trees) == 1


def test_tree_space_trees_setter():
    try:
        new_tree_space = tree.TreeSpace()
        new_tree_space.trees = 'a'
    except:
        new_tree_space = tree.TreeSpace()
        new_tree_space.trees = []

    assert len(new_tree_space.trees) == 0


def test_tree_space_best_tree():
    new_tree_space = tree.TreeSpace()

    assert isinstance(new_tree_space.best_tree, node.Node)


def test_tree_space_best_tree_setter():
    try:
        new_tree_space = tree.TreeSpace()
        new_tree_space.best_tree = 'a'
    except:
        new_tree_space = tree.TreeSpace()
        new_tree_space.best_tree = node.Node(name='0', type='FUNCTION')

    assert isinstance(new_tree_space.best_tree, node.Node)


def test_tree_space_initialize_agents():
    new_tree_space = tree.TreeSpace()

    assert new_tree_space.agents[0].position[0] != 0


def test_tree_space_initialize_terminals():
    new_tree_space = tree.TreeSpace()

    assert new_tree_space.terminals[0].position[0] != 0


def test_tree_space_create_terminals():
    new_tree_space = tree.TreeSpace(n_terminals=2)

    new_tree_space._create_terminals()

    assert len(new_tree_space.terminals) == 2


def test_tree_space_create_trees():
    new_tree_space = tree.TreeSpace(n_trees=2)

    new_tree_space._create_trees()

    assert len(new_tree_space.trees) == 2


def test_tree_space_grow():
    new_tree_space = tree.TreeSpace(min_depth=1, max_depth=5)

    new_tree = new_tree_space.grow(
        new_tree_space.min_depth, new_tree_space.max_depth)

    assert isinstance(new_tree, node.Node)
