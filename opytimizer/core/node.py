import numpy as np

import opytimizer.utils.constants as c
import opytimizer.utils.exception as e


class Node:
    """A Node instance is used for composing tree-based structures.

    """

    def __init__(self, name, type, value=None, left=None, right=None, parent=None):
        """Initialization method.

        Args:
            name (str | int): Name of the node (e.g., it should be the terminal identifier or function name).
            type (str): Type of the node (e.g., TERMINAL or FUNCTION).
            value (np.array): Value of the node (only used if it is a terminal).
            left (Node): Pointer to node's left child.
            right (Node): Pointer to node's right child.
            parent (Node): Pointer to node's parent.

        """

        # Name of the node (e.g., it should be the terminal identifier or function name)
        self.name = name

        # Type of the node (e.g., TERMINAL or FUNCTION)
        self.type = type

        # Value of the node (only if it is a terminal node)
        self.value = value

        # Pointer to node's left child
        self.left = left

        # Pointer to node's right child
        self.right = right

        # Pointer to node's parent
        self.parent = parent

        # Flag to identify whether the child is placed in the left or right
        self.flag = 1

    def __repr__(self):
        """Object representation as a formal string.

        """

        return f'{self.type}:{self.name}'

    def __str__(self):
        """Object representation as an informal string.
        """

        # Building a formatted string displaying the nodes
        lines = _build_string(self)[0]

        return '\n' + '\n'.join(lines)

    @property
    def name(self):
        """str: Node's identifier.

        """

        return self._name

    @name.setter
    def name(self, name):
        if not (isinstance(name, str) or isinstance(name, int)):
            raise e.TypeError('`name` should be a string or integer')

        self._name = name

    @property
    def type(self):
        """str: Type of the node (e.g., TERMINAL or FUNCTION).

        """

        return self._type

    @type.setter
    def type(self, type):
        if type not in ['TERMINAL', 'FUNCTION']:
            raise e.ValueError('`type` should be `TERMINAL` or `FUNCTION`')

        self._type = type

    @property
    def value(self):
        """np.array: Value of the node (only if it is a terminal node).

        """

        return self._value

    @value.setter
    def value(self, value):
        if self.type != 'TERMINAL':
            self._value = None
        else:
            if not isinstance(value, np.ndarray):
                raise e.TypeError(
                    '`value` should be an N-dimensional numpy array')

            self._value = value

    @property
    def left(self):
        """Node: Pointer to node's left child.

        """

        return self._left

    @left.setter
    def left(self, left):
        if left:
            if not isinstance(left, Node):
                raise e.TypeError('`left` should be a Node')

        self._left = left

    @property
    def right(self):
        """Node: Pointer to node's right child.

        """

        return self._right

    @right.setter
    def right(self, right):
        if right:
            if not isinstance(right, Node):
                raise e.TypeError('`right` should be a Node')

        self._right = right

    @property
    def parent(self):
        """Node: Pointer to node's parent.

        """

        return self._parent

    @parent.setter
    def parent(self, parent):
        if parent:
            if not isinstance(parent, Node):
                raise e.TypeError('`parent` should be a Node')

        self._parent = parent

    @property
    def min_depth(self):
        """int: Minimum depth of node.

        """

        return _properties(self)['min_depth']

    @property
    def max_depth(self):
        """int: Maximum depth of node.

        """

        return _properties(self)['max_depth']

    @property
    def n_leaves(self):
        """int: Number of leaves node.

        """

        return _properties(self)['n_leaves']

    @property
    def n_nodes(self):
        """int: Number of nodes.

        """

        return _properties(self)['n_nodes']

    @property
    def position(self):
        """np.array: Position after traversing the nodes.

        """

        return _evaluate(self)

    def prefix(self, node, position, flag, type, c):
        """
        """

        if node:
            c += 1
            if c == position:
                flag = node.flag
                c = 0

                if type == 'TERMINAL':
                    return node.parent

                elif node.parent.parent:
                    flag = node.parent.flag
                    return node.parent.parent

                else:
                    return None

            else:
                tmp_node = self.prefix(node.left, position, flag, type, c)
                if tmp_node:
                    return tmp_node
                else:
                    tmp_node = self.prefix(node.right, position, flag, type, c)
                    if tmp_node:
                        return tmp_node
                    else:
                        return None

        else:
            return None

    @property
    def post_order(self):

        node_stack = []
        result = []
        node = self

        while True:
            while node is not None:
                if node.right is not None:
                    node_stack.append(node.right)
                node_stack.append(node)
                node = node.left

            node = node_stack.pop()
            if (node.right is not None and
                    len(node_stack) > 0 and
                    node_stack[-1] is node.right):
                node_stack.pop()
                node_stack.append(node)
                node = node.right
            else:
                result.append(node)
                node = None

            if len(node_stack) == 0:
                break

        return result


def _build_string(node):
    """
    """

    if node is None:
        return [], 0, 0, 0

    line1 = []
    line2 = []

    node_repr = str(node.name)

    new_node_width = gap_size = len(node_repr)

    # Get the left and right sub-boxes, their widths, and node repr positions
    l_box, l_box_width, l_node_start, l_node_end = _build_string(node.left)
    r_box, r_box_width, r_node_start, r_node_end = _build_string(node.right)

    # Draw the branch connecting the current node node to the left sub-box
    # Pad the line with whitespaces where necessary
    if l_box_width > 0:
        l_node = (l_node_start + l_node_end) // 2 + 1
        line1.append(' ' * (l_node + 1))
        line1.append('_' * (l_box_width - l_node))
        line2.append(' ' * l_node + '/')
        line2.append(' ' * (l_box_width - l_node))
        new_node_start = l_box_width + 1
        gap_size += 1
    else:
        new_node_start = 0

    # Draw the representation of the current node node
    line1.append(node_repr)
    line2.append(' ' * new_node_width)

    # Draw the branch connecting the current node node to the right sub-box
    # Pad the line with whitespaces where necessary
    if r_box_width > 0:
        r_node = (r_node_start + r_node_end) // 2
        line1.append('_' * r_node)
        line1.append(' ' * (r_box_width - r_node + 1))
        line2.append(' ' * r_node + '\\')
        line2.append(' ' * (r_box_width - r_node))
        gap_size += 1
    new_node_end = new_node_start + new_node_width - 1

    # Combine the left and right sub-boxes with the branches drawn above
    gap = ' ' * gap_size
    new_box = [''.join(line1), ''.join(line2)]
    for i in range(max(len(l_box), len(r_box))):
        l_line = l_box[i] if i < len(l_box) else ' ' * l_box_width
        r_line = r_box[i] if i < len(r_box) else ' ' * r_box_width
        new_box.append(l_line + gap + r_line)

    # Return the new box, its width and its node repr positions
    return new_box, len(new_box[0]), new_node_start, new_node_end


def _evaluate(node):
    """Evaluates a node and outputs its solution array.

    Args:
        node (Node): An instance of the Node class (can be a tree of Nodes).

    Returns:
        An output solution of size (n_variables x n_dimensions).

    """

    # Checks if the node exists
    if node:
        # Performs a recursive pass on the left branch
        x = _evaluate(node.left)

        # Performs a recursive pass on the right branch
        y = _evaluate(node.right)

        # If the node is an agent or constant
        if node.type == 'TERMINAL':
            return node.value

        # If the node is a function
        else:
            # Checks if its a summation
            if node.name == 'SUM':
                return x + y

            # Checks if its a subtraction
            elif node.name == 'SUB':
                return x - y

            # Checks if its a multiplication
            elif node.name == 'MUL':
                return x * y

            # Checks if its a division
            elif node.name == 'DIV':
                return x / (y + c.EPSILON)

            # Checks if its an exponential
            elif node.name == 'EXP':
                # Checks if node is actually an array
                if type(x).__name__ == 'ndarray':
                    return np.exp(x)
                else:
                    return np.exp(y)

            # Checks if its a square root
            elif node.name == 'SQRT':
                # Checks if node is actually an array
                if type(x).__name__ == 'ndarray':
                    return np.sqrt(np.abs(x))
                else:
                    return np.sqrt(np.abs(y))

            # Checks if its a logarithm
            elif node.name == 'LOG':
                # Checks if node is actually an array
                if type(x).__name__ == 'ndarray':
                    return np.log(np.abs(x) + c.EPSILON)
                else:
                    return np.log(np.abs(y) + c.EPSILON)

            # Checks if its an absolute value
            elif node.name == 'ABS':
                # Checks if node is actually an array
                if type(x).__name__ == 'ndarray':
                    return np.abs(x)
                else:
                    return np.abs(y)

    # If the node does not exists
    else:
        return None


def _properties(root):
    """
    """

    n_nodes = 0
    n_leaves = 0
    min_depth = 0
    max_depth = -1
    current_nodes = [root]

    while len(current_nodes) > 0:
        max_depth += 1
        next_nodes = []

        for node in current_nodes:
            n_nodes += 1
            # Node is a leaf.
            if node.left is None and node.right is None:
                if min_depth == 0:
                    min_depth = max_depth
                n_leaves += 1

            if node.left is not None:
                next_nodes.append(node.left)

            if node.right is not None:
                next_nodes.append(node.right)

        current_nodes = next_nodes

    return {
        'max_depth': max_depth,
        'min_depth': min_depth,
        'n_leaves': n_leaves,
        'n_nodes': n_nodes
    }
