import sys

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.constants as c
import opytimizer.utils.logging as l
from opytimizer.core.agent import Agent
from opytimizer.core.node import Node
from opytimizer.core.space import Space

logger = l.get_logger(__name__)


class TreeSpace(Space):
    """A TreeSpace class that will hold trees, agents, variables and methods
    related to the tree-based search space.

    """

    def __init__(self, n_trees=1, n_terminals=1, n_variables=2, n_iterations=10,
                 min_depth=1, max_depth=3, functions=['SUM'],
                 lower_bound=None, upper_bound=None):
        """Initialization method.

        Args:
            n_trees (int): Number of trees.
            n_terminals (int): Number of terminal nodes.
            n_variables (int): Number of decision variables.
            n_iterations (int): Number of iterations.
            min_depth (int): Minimum depth of the trees.
            max_depth (int): Maximum depth of the trees.
            functions (list): List of functions nodes.
            lower_bound (np.array): Lower bound array with the minimum possible values.
            upper_bound (np.array): Upper bound array with the maximum possible values.

        """

        logger.info('Overriding class: Space -> TreeSpace.')

        # Override its parent class with the receiving arguments
        super(TreeSpace, self).__init__(n_agents=n_terminals, n_variables=n_variables,
                                        n_iterations=n_iterations, lower_bound=lower_bound, upper_bound=upper_bound)

        # Number of trees
        self.n_trees = n_trees

        # List of trees' fitness
        self.fit_trees = [sys.float_info.max for i in range(n_trees)]

        # Minimum depth of the trees
        self.min_depth = min_depth

        # Maximum depth of the trees
        self.max_depth = max_depth

        # List of functions nodes
        self.functions = functions

        # Now, we need to build this class up
        self._build(lower_bound, upper_bound)

        # Creating the initial trees
        self._create_trees()

        # We will log some important information
        logger.info('Class overrided.')

    def _initialize_agents(self):
        """Initialize agents' position array with uniform random numbers.

        """

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

    def _create_trees(self):
        """Creates a list of random trees using GROW algorithm.

        """

        logger.debug('Running private method: create_trees().')

        # Creates a list of random trees
        self.trees = [self.grow(self.min_depth, self.max_depth)
                      for i in range(self.n_trees)]

        logger.debug(
            f'Trees: {self.n_trees} | Depth: [{self.min_depth}, {self.max_depth}] | Functions: {self.functions}.')

    def grow(self, min_depth, max_depth):
        """It creates a random tree based on the GROW algorithm.

        References:
            S. Lukw. Two Fast Tree-Creation Algorithms for Genetic Programming. IEEE Transactions on Evolutionary Computation (2000).

        Args:
            min_depth (int): Minimum depth of the tree.
            max_depth (int): Maximum depth of the tree.

        Returns:
            A random tree based on the GROW algorithm.

        """

        # Re-initialize the agents to provide diversity
        self._initialize_agents()

        # If minimum depth equals the maximum depth
        if min_depth == max_depth:
            # Generates a terminal identifier
            terminal_id = int(
                r.generate_uniform_random_number(0, self.n_agents))

            # Return the terminal node with its id and corresponding position
            return Node(name=terminal_id, type='TERMINAL', value=self.agents[terminal_id].position)

        # If minimum depth is not equal to the maximum depth
        else:
            # Generates a node identifier
            node_id = int(r.generate_uniform_random_number(
                0, len(self.functions) + self.n_agents))

            # If the identifier is a terminal
            if node_id >= len(self.functions):
                # Gathers its real identifier
                terminal_id = node_id - len(self.functions)

                # Return the terminal node with its id and corresponding position
                return Node(name=terminal_id, type='TERMINAL', value=self.agents[terminal_id].position)

            # If the identifier is a function
            else:
                # Generates a new function node
                function_node = Node(
                    name=self.functions[node_id], type='FUNCTION')

                # For every possible function argument
                for i in range(c.N_ARGS_FUNCTION[self.functions[node_id]]):
                    # Calls recursively the grow function and creates a temporary node
                    node = self.grow(min_depth+1, max_depth)

                    # If it is not the root
                    if not i:
                        # The left child receives the temporary node
                        function_node.left = node

                    # If it is the first node
                    else:
                        # The right child receives the temporary node
                        function_node.right = node

                        # Flag to identify whether the child is in the left or right
                        node.flag = 0

                    # The parent of the temporary node is the function node
                    node.parent = function_node

                return function_node

    def get_depth(self, tree):
        """
        """

        if tree:
            return 1 + self.get_depth(tree.left) + self.get_depth(tree.right)
        else:
            return 0

    def prefix(self, tree, position, flag, type, c):
        """
        """

        if tree:
            c += 1
            if c == position:
                flag = tree.flag
                c = 0

                if type == 'TERMINAL':
                    return tree.parent

                elif tree.parent.parent:
                    flag = tree.parent.flag
                    return tree.parent.parent

                else:
                    return None

            else:
                node = self.prefix(tree.left, position, flag, type, c)
                if node:
                    return node
                else:
                    node = self.prefix(tree.right, position, flag, type, c)
                    if node:
                        return node
                    else:
                        return None

        else:
            return None
