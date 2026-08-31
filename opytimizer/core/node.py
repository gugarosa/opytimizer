"""Node."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

import opytimizer.utils.constant as c


class Node:
    """A Node instance is used for composing tree-based structures."""

    def __init__(
        self,
        name: Union[str, int],
        category: str,
        value: Optional[np.ndarray] = None,
        left: Optional[Node] = None,
        right: Optional[Node] = None,
        parent: Optional[Node] = None,
    ) -> None:
        """Initialization method.

        Args:
            name: Name of the node (e.g., it should be the terminal identifier or function name).
            category: Category of the node (e.g., TERMINAL or FUNCTION).
            value: Value of the node (only used if it is a terminal).
            left: Pointer to node's left child.
            right: Pointer to node's right child.
            parent: Pointer to node's parent.

        """

        if not isinstance(name, (str, int)):
            raise TypeError("`name` should be a string or integer")
        if category not in ("TERMINAL", "FUNCTION"):
            raise ValueError("`category` should be `TERMINAL` or `FUNCTION`")
        if category == "TERMINAL" and not isinstance(value, np.ndarray):
            raise TypeError("terminal `value` should be a numpy array")
        for label, node in (("left", left), ("right", right), ("parent", parent)):
            if node is not None and not isinstance(node, Node):
                raise TypeError(f"`{label}` should be a Node")

        self.name = name
        self.category = category
        self.value = value if category == "TERMINAL" else None

        self.left = left
        self.right = right
        self.parent = parent

        self.flag = True

    def __repr__(self) -> str:
        """Representation of a formal string."""

        return f"{self.category}:{self.name}:{self.flag}"

    @property
    def min_depth(self) -> int:
        """Minimum depth of node."""

        return _properties(self)["min_depth"]

    @property
    def max_depth(self) -> int:
        """Maximum depth of node."""

        return _properties(self)["max_depth"]

    @property
    def n_leaves(self) -> int:
        """Number of leaves node."""

        return _properties(self)["n_leaves"]

    @property
    def n_nodes(self) -> int:
        """Number of nodes."""

        return _properties(self)["n_nodes"]

    @property
    def position(self) -> np.ndarray:
        """Position after traversing the node."""

        return _evaluate(self)

    @property
    def post_order(self) -> List[Node]:
        """Traverses the node in post-order."""

        post_order, stacked = [], []

        while True:
            while self is not None:
                if self.right is not None:
                    stacked.append(self.right)

                stacked.append(self)

                self = self.left

            self = stacked.pop()

            if (
                self.right is not None
                and len(stacked) > 0
                and stacked[-1] is self.right
            ):
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
    def pre_order(self) -> List[Node]:
        """Traverses the node in pre-order."""

        pre_order, stacked = [], [self]

        while len(stacked) > 0:
            node = stacked.pop()
            pre_order.append(node)

            if node.right is not None:
                stacked.append(node.right)

            if node.left is not None:
                stacked.append(node.left)

        return pre_order

    def find_node(self, position: int) -> Tuple[Optional[Node], bool]:
        """Finds a node at a given position.

        Args:
            position: Position of the node.

        Returns:
            (Node): Node at desired position.

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


def _evaluate(node: Node) -> np.ndarray:
    """Evaluates a node and outputs its solution array.

    Args:
        node: An instance of the Node class (can be a tree of Nodes).

    Returns:
        (np.ndarray): Output solution of size (n_variables x n_dimensions).

    """

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


def _properties(node: Node) -> Dict[str, Any]:
    """Traverses the node and returns some useful properties.

    Args:
        node: An instance of the Node class (can be a tree of Nodes).

    Returns:
        (Dict[str, Any]): Dictionary containing some useful properties: `min_depth`, `max_depth`,
        `n_leaves` and `n_nodes`.

    """

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
