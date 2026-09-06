# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Node.

"""

from __future__ import annotations

import numpy as np

import opytimizer.utils.constant as c


class Node:
    """A Node instance is used for composing tree-based structures.

    """

    def __init__(
        self,
        name: str | int,
        category: str,
        value: np.ndarray | None = None,
        left: Node | None = None,
        right: Node | None = None,
        parent: Node | None = None,
    ) -> None:
        """Initialization method.

        Args:
            name: Name of the node (e.g., it should be the terminal identifier or function name).
            category: Category of the node (e.g., TERMINAL or FUNCTION).
            value: Value of the node (only used if it is a terminal).
            left: Pointer to node's left child.
            right: Pointer to node's right child.
            parent: Pointer to node's parent.

        Raises:
            TypeError: The name, terminal value, or related node reference has an invalid type.
            ValueError: The category is neither ``TERMINAL`` nor ``FUNCTION``.

        Notes:
            Child and parent references are retained as supplied, not linked
            automatically. Terminal values share their input arrays. Function
            nodes ignore ``value`` and evaluate their children on each access
            to ``position``.

        """

        if not isinstance(name, (str, int)):
            raise TypeError("`name` should be a string or integer.")
        if category not in ("TERMINAL", "FUNCTION"):
            raise ValueError("`category` should be `TERMINAL` or `FUNCTION`.")
        if category == "TERMINAL" and not isinstance(value, np.ndarray):
            raise TypeError("`value` should be a numpy array for a terminal.")
        for label, node in (("left", left), ("right", right), ("parent", parent)):
            if node is not None and not isinstance(node, Node):
                raise TypeError(f"`{label}` should be a Node.")

        self.name = name
        self.category = category
        self.value = value if category == "TERMINAL" else None

        self.left = left
        self.right = right
        self.parent = parent

        self.flag = True

    def __repr__(self) -> str:
        """Represent the node category, name, and branch flag.

        """

        return f"{self.category}:{self.name}:{self.flag}"

    @property
    def min_depth(self) -> int:
        """Minimum depth of node.

        """

        return _properties(self)["min_depth"]

    @property
    def max_depth(self) -> int:
        """Maximum depth of node.

        """

        return _properties(self)["max_depth"]

    @property
    def n_leaves(self) -> int:
        """Number of leaf nodes.

        """

        return _properties(self)["n_leaves"]

    @property
    def n_nodes(self) -> int:
        """Number of nodes.

        """

        return _properties(self)["n_nodes"]

    @property
    def position(self) -> np.ndarray:
        """Evaluate the tree into a position array without copying terminal values.

        """

        return _evaluate(self)

    @property
    def post_order(self) -> list[Node]:
        """Traverses the node in post-order.

        """

        post_order, stacked = [], []

        while True:
            while self is not None:
                if self.right is not None:
                    stacked.append(self.right)

                stacked.append(self)

                self = self.left

            self = stacked.pop()

            if self.right is not None and len(stacked) > 0 and stacked[-1] is self.right:
                stacked.pop()
                stacked.append(self)

                self = self.right
            else:
                post_order.append(self)

                self = None

            if len(stacked) == 0:
                break

        return post_order

    @property
    def pre_order(self) -> list[Node]:
        """Traverses the node in pre-order.

        """

        pre_order, stacked = [], [self]

        while len(stacked) > 0:
            node = stacked.pop()
            pre_order.append(node)

            if node.right is not None:
                stacked.append(node.right)

            if node.left is not None:
                stacked.append(node.left)

        return pre_order

    def find_node(self, position: int) -> tuple[Node | None, bool]:
        """Find the parent insertion point associated with a pre-order position.

        Args:
            position: Index in the pre-order traversal.

        Returns:
            Ancestor node and left-child flag, or ``(None, False)`` when no insertion point exists.

        """

        pre_order = self.pre_order
        if len(pre_order) > position:
            node = pre_order[position]

            if node.category == "TERMINAL":
                return node.parent, node.flag

            if node.category == "FUNCTION":
                if node.parent and node.parent.parent:
                    return node.parent.parent, node.parent.flag

                return None, False

        return None, False


def _evaluate(node: Node | None) -> np.ndarray | None:
    if node:
        x = _evaluate(node.left)
        y = _evaluate(node.right)

        if node.category == "TERMINAL":
            return node.value

        if node.name == "SUM":
            return x + y

        if node.name == "SUB":
            return x - y

        if node.name == "MUL":
            return x * y

        if node.name == "DIV":
            return x / (y + c.EPSILON)

        if node.name == "EXP":
            return np.exp(x)

        if node.name == "SQRT":
            return np.sqrt(np.abs(x))

        if node.name == "LOG":
            return np.log(np.abs(x) + c.EPSILON)

        if node.name == "ABS":
            return np.abs(x)

        if node.name == "SIN":
            return np.sin(x)

        if node.name == "COS":
            return np.cos(x)

    return None


def _properties(node: Node) -> dict[str, int]:
    min_depth, max_depth = 0, -1
    n_leaves = n_nodes = 0

    nodes = [node]
    while len(nodes) > 0:
        max_depth += 1

        next_nodes = []
        for n in nodes:
            n_nodes += 1

            if n.left is None and n.right is None:
                if min_depth == 0:
                    min_depth = max_depth

                n_leaves += 1

            if n.left is not None:
                next_nodes.append(n.left)

            if n.right is not None:
                next_nodes.append(n.right)

        nodes = next_nodes

    return {
        "min_depth": min_depth,
        "max_depth": max_depth,
        "n_leaves": n_leaves,
        "n_nodes": n_nodes,
    }
