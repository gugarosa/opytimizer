import sys

import numpy as np

import opytimizer.math.constants as c
import opytimizer.math.random as r
import opytimizer.utils.logging as l
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space
from opytimizer.utils.node import Node

logger = l.get_logger(__name__)


class TreeSpace(Space):
    """A TreeSpace class that will hold trees, agents, variables and methods
    related to the tree-based search space.

    """

    def __init__(self, n_trees=1, n_variables=2, n_iterations=10,
                 min_depth=1, max_depth=1, functions=['SUM'], terminals=['AGENT', 'CONSTANT'],
                 lower_bound=None, upper_bound=None):
        """Initialization method.

        Args:
            n_trees (int): Number of trees.
            n_variables (int): Number of decision variables.
            n_iterations (int): Number of iterations.
            min_depth (int): Minimum depth of the trees.
            max_depth (int): Maximum depth of the trees.
            functions (list): List of functions nodes.
            terminals (list): List of terminal nodes.
            lower_bound (np.array): Lower bound array with the minimum possible values.
            upper_bound (np.array): Upper bound array with the maximum possible values.

        """

        logger.info('Overriding class: Space -> TreeSpace.')

        # Override its parent class with the receiving arguments
        super(TreeSpace, self).__init__(n_agents=len(terminals), n_variables=n_variables,
                                        n_iterations=n_iterations, lower_bound=lower_bound, upper_bound=upper_bound)

        # Number of trees
        self.n_trees = n_trees

        # List of trees
        self.trees = None

        # List of trees' fitness
        self.fit_trees = [sys.float_info.max for i in range(n_trees)]

        # Minimum depth of the trees
        self.min_depth = min_depth

        # Maximum depth of the trees
        self.max_depth = max_depth

        # List of functions nodes
        self.functions = functions

        # List of terminal nodes
        self.terminals = terminals

        # Now, we need to build this class up
        self._build(lower_bound, upper_bound)

        logger.debug(
            f'Trees: {self.n_trees} | Depth: [{self.min_depth}, {self.max_depth}] | Functions: {self.functions} | Terminals: {self.terminals} | Built: {self.built}.')

        # Initializing agents
        self._initialize_agents()

        # Initializing trees
        self._initialize_trees()

        # We will log some important information
        logger.info('Class overrided.')

    def _grow(self, min_depth, max_depth):
        """It creates a random tree based on the GROW algorithm.

        References:
            S. Lukw. Two Fast Tree-Creation Algorithms for Genetic Programming. IEEE Transactions on Evolutionary Computation (2000).

        Args:
            min_depth (int): Minimum depth of the tree.
            max_depth (int): Maximum depth of the tree.

        Returns:
            A random tree based on the GROW algorithm.

        """

        # Defining the possible function arguments dictionary
        args = {
            'SUM': 2,
            'SUB': 2,
            'MUL': 2,
            'DIV': 2,
            'EXP': 1,
            'SQRT': 1,
            'LOG': 1,
            'ABS': 1
        }

        # If minimum depth equals the maximum depth
        if min_depth == max_depth:
            # Generates the identifier of the terminal
            id = int(r.generate_uniform_random_number(0, len(self.terminals)))

            # If the terminal is a constant
            if self.terminals[id] == 'CONSTANT':
                # Returns a constant node
                return Node(id=id, name=self.terminals[id], type='CONSTANT')

            # If the terminal is a terminal, returns an agent node
            return Node(id=id, name=self.terminals[id], type='AGENT')

        # If minimum depth is not equal to the maximum depth
        else:
            # Generates the identifier of the terminal
            id = int(r.generate_uniform_random_number(
                0, len(self.functions) + len(self.terminals)))

            # If the identifier is a terminal
            if id >= len(self.functions):
                # Gathers its real identifier
                id -= len(self.functions)

                # If the terminal is a constant
                if self.terminals[id] == 'CONSTANT':
                    # Returns a constant node
                    return Node(id=id, name=self.terminals[id], type='CONSTANT')

                # If the terminal is a terminal, returns an agent node
                return Node(id=id, name=self.terminals[id], type='AGENT')

            # If the identifier is a function
            else:
                # Generates a new node
                node = Node(id=id, name=self.functions[id], type='FUNCTION')

                # For every possible function argument
                for i in range(args[self.functions[id]]):
                    # Calls recursively the grow function
                    tmp_node = self._grow(min_depth+1, max_depth)

                    # If it is not the first node
                    if not i:
                        # The left child receives the temporary node
                        node.left = tmp_node

                    # If it is the first node
                    else:
                        # The right chield receives the temporary node
                        node.right = tmp_node

                    # The parent of the temporary node is the node
                    tmp_node.parent = node

                return node

    def _initialize_agents(self):
        """Initialize agents' position array with uniform random numbers.

        """

        logger.debug('Running private method: initialize_agents().')

        # Iterate through all agents
        for agent in self.agents:
            # Iterate through all decision variables
            for j, (lb, ub) in enumerate(zip(self.lb, self.ub)):
                # For each decision variable, we generate uniform random numbers
                agent.position[j] = r.generate_uniform_random_number(
                    lb, ub, size=agent.n_dimensions)

                # For each decision variable, we apply lower bound the agent's bound
                agent.lb[j] = lb

                # And also the upper bound
                agent.ub[j] = ub

        logger.debug('Agents initialized.')

    def _initialize_trees(self):
        """Initialize a list of random created trees.

        """

        logger.debug('Running private method: initialize_trees().')

        # Creates a list of random trees based on the GROW algorithm
        self.trees = [self._grow(self.min_depth, self.max_depth)
                      for i in range(self.n_trees)]

        logger.debug('Trees initialized.')

    def run_tree(self, tree):
        """Runs the tree and outputs its solution array.

        Args:
            tree (Node): A tree structure based on the Node class.

        Returns:
            An output solution of size (n_variables x n_dimensions).

        """

        # Checks if the tree exists
        if tree:
            # Performs a recursive pass on the left branch
            x = self.run_tree(tree.left)

            # Performs a recursive pass on the right branch
            y = self.run_tree(tree.right)

            # If the node is an agent or constant
            if tree.type == 'AGENT' or tree.type == 'CONSTANT':
                return self.agents[tree.id].position

            # If the node is a function
            else:
                # Checks if its a summation
                if tree.name == 'SUM':
                    return x + y

                # Checks if its a subtraction
                elif tree.name == 'SUB':
                    return x - y

                # Checks if its a multiplication
                elif tree.name == 'MUL':
                    return x * y

                # Checks if its a division
                elif tree.name == 'DIV':
                    return x / (y + c.EPSILON)

                # Checks if its an exponential
                elif tree.name == 'EXP':
                    # Checks if node is actually an array
                    if type(x).__name__ == 'ndarray':
                        return np.exp(x)
                    else:
                        return np.exp(y)

                # Checks if its a square root
                elif tree.name == 'SQRT':
                    # Checks if node is actually an array
                    if type(x).__name__ == 'ndarray':
                        return np.sqrt(np.abs(x))
                    else:
                        return np.sqrt(np.abs(y))

                # Checks if its a logarithm
                elif tree.name == 'LOG':
                    # Checks if node is actually an array
                    if type(x).__name__ == 'ndarray':
                        return np.log(np.abs(x) + c.EPSILON)
                    else:
                        return np.log(np.abs(y) + c.EPSILON)

                # Checks if its an absolute value
                elif tree.name == 'ABS':
                    # Checks if node is actually an array
                    if type(x).__name__ == 'ndarray':
                        return np.abs(x)
                    else:
                        return np.abs(y)

        # If the tree does not exists
        else:
            return None
