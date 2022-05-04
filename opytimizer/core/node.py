"""Node.
"""

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
        left: Optional["Node"] = None,
        right: Optional["Node"] = None,
        parent: Optional["Node"] = None,
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

        # Name of the node (terminal identifier or function name)
        self.name = name

        # Category of the node (`TERMINAL` or `FUNCTION`)
        self.category = category

        # Value of the node (only for terminal nodes)
        self.value = value

        # Pointers to the node's children and parent
        self.left = left
        self.right = right
        self.parent = parent

        # Flag to identify whether the node is a left child
        self.flag = True

    def __repr__(self) -> str:
        """Representation of a formal string."""

        return f"{self.category}:{self.name}:{self.flag}"

    def __str__(self) -> str:
        """Representation of an informal string."""

        # Building a formatted string for displaying the nodes
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
    def left(self) -> "Node":
        """Pointer to the node's left child."""

        return self._left

    @left.setter
    def left(self, left: "Node") -> None:
        if left and not isinstance(left, Node):
            raise e.TypeError("`left` should be a Node")

        self._left = left

    @property
    def right(self) -> "Node":
        """Pointer to the node's right child."""

        return self._right

    @right.setter
    def right(self, right: "Node") -> None:
        if right and not isinstance(right, Node):
            raise e.TypeError("`right` should be a Node")

        self._right = right

    @property
    def parent(self) -> "Node":
        """Pointer to the node's parent."""

        return self._parent

    @parent.setter
    def parent(self, parent: "Node") -> None:
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
    def post_order(self) -> List["Node"]:
        """Traverses the node in post-order."""

        # Creates lists for post-order and stacked nodes
        post_order, stacked = [], []

        while True:
            # Creates another while to check if node exists
            while self is not None:
                # If there is a right child node
                if self.right is not None:
                    # Appends the right child node
                    stacked.append(self.right)

                # Appends the current node
                stacked.append(self)

                # Gathers the left child node
                self = self.left

            # Pops the stacked nodes
            self = stacked.pop()

            # If there is a right node, stacked nodes and the last stacked was a right child
            if (
                self.right is not None
                and len(stacked) > 0
                and stacked[-1] is self.right
            ):
                # Pops the stacked node
                stacked.pop()

                # Appends current node
                stacked.append(self)

                # Gathers the right child node
                self = self.right

            else:
                # Appends the node to the output list
                post_order.append(self)

                # And apply None as the current node
                self = None

            if len(stacked) == 0:
                break

        return post_order

    @property
    def pre_order(self) -> List["Node"]:
        """Traverses the node in pre-order."""

        # Creates lists for pre-order and stacked nodes
        pre_order, stacked = [], [self]

        # While there is more than one node
        while len(stacked) > 0:
            # Pops the list and gets the node
            node = stacked.pop()

            # Appends to the pre-order
            pre_order.append(node)

            # If there is a child in the right
            if node.right is not None:
                # Appends the child
                stacked.append(node.right)

            # If there is a child in the left
            if node.left is not None:
                # Appends the child
                stacked.append(node.left)

        return pre_order

    def find_node(self, position: int) -> "Node":
        """Finds a node at a given position.

        Args:
            position: Position of the node.

        Returns:
            (Node): Node at desired position.

        """

        # Calculates the pre-order of current node
        pre_order = self.pre_order

        # Checks if the pre-order list has more nodes than the desired position
        if len(pre_order) > position:
            # Gets the node from position
            node = pre_order[position]

            if node.category == "TERMINAL":
                return node.parent, node.flag

            if node.category == "FUNCTION":
                # If it is a function node, we need to return the parent of its parent
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
        # Return an empty list along with `0` arguments
        return [], 0, 0, 0

    # Creates lists to hold the first and second lines
    first_line, second_line = [], []

    # Gets the node name as a string
    name = str(node.name)

    # The gap size and width of the new node will be the length of the name's string
    gap = width = len(name)

    # Iterate recursively through the left and right branches
    left_branch, left_width, left_start, left_end = _build_string(node.left)
    right_branch, right_width, right_start, right_end = _build_string(node.right)

    if left_width > 0:
        # Calculates the left node
        left = (left_start + left_end) // 2 + 1

        # Appends to first line space and underscore chars
        first_line.append(" " * (left + 1))
        first_line.append("_" * (left_width - left))

        # Appends to second line space chars and connecting slash
        second_line.append(" " * left + "/")
        second_line.append(" " * (left_width - left))

        # The start point will be the left width plus one
        start = left_width + 1

        # Increases the gap
        gap += 1

    else:
        # The start point will be 0
        start = 0

    # Appending current node's name to first line
    first_line.append(name)

    # Appending space chars to second line based on the node's width
    second_line.append(" " * width)

    if right_width > 0:
        # Calculates the right node
        right = (right_start + right_end) // 2

        # Appends to first line underscore and space chars
        first_line.append("_" * right)
        first_line.append(" " * (right_width - right + 1))

        # Appends to second line space chars and a connecting backslash
        second_line.append(" " * right + "\\")
        second_line.append(" " * (right_width - right))

        # Increases the gap size
        gap += 1

    # The ending point will be start plus width minus 1
    end = start + width - 1

    # Calculates how many gaps are needed
    gap = " " * gap

    # Combining left and right branches
    lines = ["".join(first_line), "".join(second_line)]

    # For every possible value in the branches
    for i in range(max(len(left_branch), len(right_branch))):
        # If current iteration is smaller than left branch's size
        if i < len(left_branch):
            # Applies the left branch to the left line
            left_line = left_branch[i]

        else:
            # Apply space chars
            left_line = " " * left_width

        # If current iteration is smaller than right branch's size
        if i < len(right_branch):
            # Applies the right branch to the right line
            right_line = right_branch[i]

        else:
            # Apply space chars
            right_line = " " * right_width

        # Appends the whole line
        lines.append(left_line + gap + right_line)

    # Return the new box, its width and its node repr positions
    return lines, len(lines[0]), start, end


def _evaluate(node: Node) -> np.ndarray:
    """Evaluates a node and outputs its solution array.

    Args:
        node: An instance of the Node class (can be a tree of Nodes).

    Returns:
        (np.ndarray): Output solution of size (n_variables x n_dimensions).

    """

    if node:
        # Performs a recursive pass on the left and right branches
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

    # Initializes minimum and maximum depths
    min_depth, max_depth = 0, -1

    # Initializes number of leaves and nodes as 0
    n_leaves = n_nodes = 0

    # Gathers a list of possible nodes
    nodes = [node]

    while len(nodes) > 0:
        # Maximum depth increases by 1
        max_depth += 1

        # Creates a list for further nodes
        next_nodes = []

        for n in nodes:
            # Increases the number of nodes
            n_nodes += 1

            # If the node is a leaf
            if n.left is None and n.right is None:
                if min_depth == 0:
                    # Minimum depth will be equal to maximum depth
                    min_depth = max_depth

                # Increases the number of leaves by 1
                n_leaves += 1

            # If there is a child in the left
            if n.left is not None:
                # Appends the left child node
                next_nodes.append(n.left)

            # If there is a child in the right
            if n.right is not None:
                # Appends the right child node
                next_nodes.append(n.right)

        # Current nodes will receive the list of the next depth
        nodes = next_nodes

    return {
        "min_depth": min_depth,
        "max_depth": max_depth,
        "n_leaves": n_leaves,
        "n_nodes": n_nodes,
    }
