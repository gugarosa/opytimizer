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

        # Flag to identify whether the node is a left child
        self.flag = True

    def __repr__(self):
        """Object representation as a formal string.

        """

        return f'{self.type}:{self.name}:{self.flag}'

    def __str__(self):
        """Object representation as an informal string.

        """

        # Building a formatted string for displaying the nodes
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
    def flag(self):
        """bool: Flag to identify whether the node is a left child.

        """

        return self._flag

    @flag.setter
    def flag(self, flag):
        if not isinstance(flag, bool):
            raise e.TypeError('`flag` should be a boolean')

        self._flag = flag

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

    @property
    def post_order(self):
        """list: Traverses the nodes in post-order.

        """

        # Creates a list for outputting the nodes
        post_order = []

        # Creates a list to hold the stacked nodes
        stacked = []

        # Creates a perpetual while
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
            if (self.right is not None and len(stacked) > 0 and stacked[-1] is self.right):
                # Pops the stacked node
                stacked.pop()

                # Appends current node
                stacked.append(self)

                # Gathers the right child node
                self = self.right

            # If the condition fails
            else:
                # Appends the node to the output list
                post_order.append(self)

                # And apply None as the current node
                self = None

            # If the stacked list is empty
            if len(stacked) == 0:
                # Breaks the loop
                break

        return post_order

    @property
    def pre_order(self):
        """list: Traverses the nodes in pre-order.

        """

        # Creates a list for outputting the nodes
        pre_order = []

        # Creates a list to hold the stacked nodes
        stacked = [self]

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

    def find_node(self, position):
        """Finds a node at a given position.

        Args:
            position (int): Position of the node.

        Returns:
            The node at desired position.

        """

        # Calculates the pre-order of current node
        pre_order = self.pre_order

        # Checks if the pre-order list has more nodes than the desired position
        if len(pre_order) > position:
            # Gets the node from position
            node = pre_order[position]

            # If the node is a terminal
            if node.type == 'TERMINAL':
                return node.parent, node.flag

            # If the node is a function
            elif node.type == 'FUNCTION':
                # If it is a function node, we need to return the parent of its parent
                if node.parent.parent:
                    return node.parent.parent, node.parent.flag

                # If there is no parent of parent
                else:
                    return None, False

        return None, False


def _build_string(node):
    """Builds a formatted string for displaying the nodes.

    References:
        https://github.com/joowani/binarytree/blob/master/binarytree/__init__.py#L153

    Args:
        node (Node): An instance of the Node class (can be a tree of Nodes).

    Returns:
        An output solution of size (n_variables x n_dimensions).

    """

    # If current node is None
    if node is None:
        # Return an empty list along with `0` arguments
        return [], 0, 0, 0

    # Creates a list to hold the first line
    first_line = []

    # And also a list to hold the second line
    second_line = []

    # Gets the node name as a string
    name = str(node.name)

    # The gap size and width of the new node will be the length of the name's string
    gap = width = len(name)

    # Iterate recursively through the left branch
    left_branch, left_width, left_start, left_end = _build_string(node.left)

    # Iterate recursively through the right branch
    right_branch, right_width, right_start, right_end = _build_string(
        node.right)

    # If left branch width is greater than 0
    if left_width > 0:
        # Calculates the left node
        left = (left_start + left_end) // 2 + 1

        # Appends to first line space chars
        first_line.append(' ' * (left + 1))

        # Appends to first line underscore chars
        first_line.append('_' * (left_width - left))

        # Appends to second line space chars and a connecting slash
        second_line.append(' ' * left + '/')

        # Appends to second line space chars
        second_line.append(' ' * (left_width - left))

        # The start point will be the left width plus one
        start = left_width + 1

        # Increases the gap
        gap += 1

    # If not
    else:
        # The start point will be 0
        start = 0

    # Appending current node's name to first line
    first_line.append(name)

    # Appending space chars to second line based on the node's width
    second_line.append(' ' * width)

    # If right branch width is greater than 0
    if right_width > 0:
        # Calculates the right node
        right = (right_start + right_end) // 2

        # Appends to first line underscore chars
        first_line.append('_' * right)

        # Appends to first line space chars
        first_line.append(' ' * (right_width - right + 1))

        # Appends to second line space chars and a connecting backslash
        second_line.append(' ' * right + '\\')

        # Appends to second line space chars
        second_line.append(' ' * (right_width - right))

        # Increases the gap size
        gap += 1

    # The ending point will be start plus width minus 1
    end = start + width - 1

    # Calculates how many gaps are needed
    gap = ' ' * gap

    # Combining left and right branches
    lines = [''.join(first_line), ''.join(second_line)]

    # For every possible value in the branches
    for i in range(max(len(left_branch), len(right_branch))):
        # If current iteration is smaller than left branch's size
        if i < len(left_branch):
            # Applies the left branch to the left line
            left_line = left_branch[i]

        # If not
        else:
            # Apply space chars
            left_line = ' ' * left_width

        # If current iteration is smaller than right branch's size
        if i < len(right_branch):
            # Applies the right branch to the right line
            right_line = right_branch[i]

        # If not
        else:
            # Apply space chars
            right_line = ' ' * right_width

        # Appends the whole line
        lines.append(left_line + gap + right_line)

    # Return the new box, its width and its node repr positions
    return lines, len(lines[0]), start, end


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
                return np.exp(x)

            # Checks if its a square root
            elif node.name == 'SQRT':
                return np.sqrt(np.abs(x))

            # Checks if its a logarithm
            elif node.name == 'LOG':
                return np.log(np.abs(x) + c.EPSILON)

            # Checks if its an absolute value
            elif node.name == 'ABS':
                return np.abs(x)

            # Checks if its a sine value
            elif node.name == 'SIN':
                return np.sin(x)

            # Checks if its a cosine value
            elif node.name == 'COS':
                return np.cos(x)

    # If the node does not exists
    else:
        return None


def _properties(node):
    """Traverses the nodes and returns some useful properties.

    Args:
        node (Node): An instance of the Node class (can be a tree of Nodes).

    Returns:
        A dictionary containing some useful properties: `min_depth`, `max_depth`,
        `n_leaves` and `n_nodes`.

    """

    # Initializing minimum depth as 0
    min_depth = 0

    # Initializing maximum depth as -1
    max_depth = -1

    # Initializing number of leaves and nodes as 0
    n_leaves = n_nodes = 0

    # Gathering a list of possible nodes
    nodes = [node]

    # While there is a nonde
    while len(nodes) > 0:
        # Maximum depth increases by 1
        max_depth += 1

        # Creates a list for further nodes
        next_nodes = []

        # For each node in the current ones
        for node in nodes:
            # Increases the number of nodes
            n_nodes += 1

            # If the node is a leaf
            if node.left is None and node.right is None:
                # If minimum depth is equal to 0
                if min_depth == 0:
                    # Minimum depth will be equal to maximum depth
                    min_depth = max_depth

                # Increases the number of leaves by 1
                n_leaves += 1

            # If there is a child in the left
            if node.left is not None:
                # Appends the left child node
                next_nodes.append(node.left)

            # If there is a child in the right
            if node.right is not None:
                # Appends the right child node
                next_nodes.append(node.right)

        # Current nodes will receive the list of the next depth
        nodes = next_nodes

    return {
        'min_depth': min_depth,
        'max_depth': max_depth,
        'n_leaves': n_leaves,
        'n_nodes': n_nodes
    }
