import sys

import opytimizer.utils.constants as c

class Node:
    """
    """

    def __init__(self, name, type, value=None, right=None, left=None, parent=None):
        """Initialization method.

        Args:
            name (int | str):
            type (str):
            value (np.array):
            right (Node):
            
        """

        # Name of the node (e.g., it should be the terminal identifier or function name)
        self.name = name

        # Type of the node (e.g., TERMINAL or FUNCTION)
        self.type = type

        # Value of the node (only if it is a terminal node)
        self.value = value

        # Pointer to node's right child
        self.right = right

        # Pointer to node's left child
        self.left = left

        # Flag to identify whether the child is placed in the left or right
        self.flag = 1

        # Pointer to node's parents
        self.parent = parent

    def _build_tree_string(self, root, curr_index, index=False, delimiter='-'):
        if root is None:
            return [], 0, 0, 0

        line1 = []
        line2 = []
        if index:
            node_repr = '{}{}{}'.format(curr_index, delimiter, root.name)
        else:
            node_repr = str(root.name)

        new_root_width = gap_size = len(node_repr)

        # Get the left and right sub-boxes, their widths, and root repr positions
        l_box, l_box_width, l_root_start, l_root_end = self._build_tree_string(root.left, 2 * curr_index + 1, index, delimiter)
        r_box, r_box_width, r_root_start, r_root_end = self._build_tree_string(root.right, 2 * curr_index + 2, index, delimiter)

        # Draw the branch connecting the current root node to the left sub-box
        # Pad the line with whitespaces where necessary
        if l_box_width > 0:
            l_root = (l_root_start + l_root_end) // 2 + 1
            line1.append(' ' * (l_root + 1))
            line1.append('_' * (l_box_width - l_root))
            line2.append(' ' * l_root + '/')
            line2.append(' ' * (l_box_width - l_root))
            new_root_start = l_box_width + 1
            gap_size += 1
        else:
            new_root_start = 0

        # Draw the representation of the current root node
        line1.append(node_repr)
        line2.append(' ' * new_root_width)

        # Draw the branch connecting the current root node to the right sub-box
        # Pad the line with whitespaces where necessary
        if r_box_width > 0:
            r_root = (r_root_start + r_root_end) // 2
            line1.append('_' * r_root)
            line1.append(' ' * (r_box_width - r_root + 1))
            line2.append(' ' * r_root + '\\')
            line2.append(' ' * (r_box_width - r_root))
            gap_size += 1
        new_root_end = new_root_start + new_root_width - 1

        # Combine the left and right sub-boxes with the branches drawn above
        gap = ' ' * gap_size
        new_box = [''.join(line1), ''.join(line2)]
        for i in range(max(len(l_box), len(r_box))):
            l_line = l_box[i] if i < len(l_box) else ' ' * l_box_width
            r_line = r_box[i] if i < len(r_box) else ' ' * r_box_width
            new_box.append(l_line + gap + r_line)

        # Return the new box, its width and its root repr positions
        return new_box, len(new_box[0]), new_root_start, new_root_end

    def _properties(self, root):
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

    def _evaluate(self, node):
        """Evaluates a node and outputs its solution array.

        Args:
            node (Node): An instance of the Node class (can be a tree of Nodes).

        Returns:
            An output solution of size (n_variables x n_dimensions).

        """

        # Checks if the node exists
        if node:
            # Performs a recursive pass on the left branch
            x = self._evaluate(node.left)

            # Performs a recursive pass on the right branch
            y = self._evaluate(node.right)

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
        

    def __repr__(self):
        return f'{self.type}:{self.name}'

    def __str__(self):
        lines = self._build_tree_string(self, 0, False, '-')[0]
        return '\n' + '\n'.join((line.rstrip() for line in lines))

    @property
    def max_depth(self):
        return self._properties(self)['max_depth']

    @property
    def min_depth(self):
        return self._properties(self)['min_depth']

    @property
    def n_leaves(self):
        return self._properties(self)['n_leaves']

    @property
    def n_nodes(self):
        return self._properties(self)['n_nodes']

    @property
    def position(self):
        return self._evaluate(self)

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
