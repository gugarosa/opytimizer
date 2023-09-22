"""Node.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np

import opytimizer.utils.constant as c
import opytimizer.utils.exception as e


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

        self.name = name
        self.category = category
        self.value = value

        self.left = left
        self.right = right
        self.parent = parent

        self.flag = True

    def __repr__(self) -> str:
        """Representation of a formal string."""

        return f"{self.category}:{self.name}:{self.flag}"

    def __str__(self) -> str:
        """Representation of an informal string."""

        lines = _build_string(self)[0]

        return "\n" + "\n".join(lines)

    @property
    def name(self) -> Union[str, int]:
        """Name of the node."""

        return self._name

    @name.setter
    def name(self, name: Union[str, int]) -> None:
        if not isinstance(name, (str, int)):
            raise e.TypeError("`name` should be a string or integer")

        self._name = name

    @property
    def category(self) -> str:
        """Category of the node."""

        return self._category

    @category.setter
    def category(self, category: str) -> None:
        if category not in ["TERMINAL", "FUNCTION"]:
            raise e.ValueError("`category` should be `TERMINAL` or `FUNCTION`")

        self._category = category

    @property
    def value(self) -> np.ndarray:
        """np.array: Value of the node."""

        return self._value

    @value.setter
    def value(self, value: np.ndarray) -> None:
        if self.category != "TERMINAL":
            self._value = None
        else:
            if not isinstance(value, np.ndarray):
                raise e.TypeError("`value` should be an N-dimensional numpy array")

            self._value = value

    @property
    def left(self) -> Node:
        """Pointer to the node's left child."""

        return self._left

    @left.setter
    def left(self, left: Node) -> None:
        if left and not isinstance(left, Node):
            raise e.TypeError("`left` should be a Node")

        self._left = left

    @property
    def right(self) -> Node:
        """Pointer to the node's right child."""

        return self._right

    @right.setter
    def right(self, right: Node) -> None:
        if right and not isinstance(right, Node):
            raise e.TypeError("`right` should be a Node")

        self._right = right

    @property
    def parent(self) -> Node:
        """Pointer to the node's parent."""

        return self._parent

    @parent.setter
    def parent(self, parent: Node) -> None:
        if parent and not isinstance(parent, Node):
            raise e.TypeError("`parent` should be a Node")

        self._parent = parent

    @property
    def flag(self) -> bool:
        """Flag to identify whether the node is a left child."""

        return self._flag

    @flag.setter
    def flag(self, flag: bool) -> None:
        if not isinstance(flag, bool):
            raise e.TypeError("`flag` should be a boolean")

        self._flag = flag

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

    def find_node(self, position: int) -> Node:
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


def _build_string(node: Node) -> str:
    """Builds a formatted string for displaying the nodes.

    References:
        https://github.com/joowani/binarytree/blob/master/binarytree/__init__.py#L153

    Args:
        node: An instance of the Node class (can be a tree of Nodes).

    Returns:
        (str): Formatted string ready to be printed.

    """

    if node is None:
        return [], 0, 0, 0

    first_line, second_line = [], []

    name = str(node.name)
    gap = width = len(name)

    left_branch, left_width, left_start, left_end = _build_string(node.left)
    right_branch, right_width, right_start, right_end = _build_string(node.right)

    if left_width > 0:
        left = (left_start + left_end) // 2 + 1

        first_line.append(" " * (left + 1))
        first_line.append("_" * (left_width - left))

        second_line.append(" " * left + "/")
        second_line.append(" " * (left_width - left))

        start = left_width + 1
        gap += 1
    else:
        start = 0

    first_line.append(name)
    second_line.append(" " * width)

    if right_width > 0:
        right = (right_start + right_end) // 2

        first_line.append("_" * right)
        first_line.append(" " * (right_width - right + 1))

        second_line.append(" " * right + "\\")
        second_line.append(" " * (right_width - right))

        gap += 1

    end = start + width - 1
    gap = " " * gap

    lines = ["".join(first_line), "".join(second_line)]

    for i in range(max(len(left_branch), len(right_branch))):
        if i < len(left_branch):
            left_line = left_branch[i]
        else:
            left_line = " " * left_width

        if i < len(right_branch):
            right_line = right_branch[i]
        else:
            right_line = " " * right_width

        lines.append(left_line + gap + right_line)

    return lines, len(lines[0]), start, end


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
