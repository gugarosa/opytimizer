import copy

import numpy as np

import opytimizer.math.random as r
import opytimizer.utils.constants as c
import opytimizer.utils.exception as e
import opytimizer.utils.logging as l
from opytimizer.core.agent import Agent
from opytimizer.core.node import Node
from opytimizer.core.space import Space

logger = l.get_logger(__name__)


class TreeSpace(Space):
    """A TreeSpace class for trees, agents, variables and methods
    related to a tree-based search space.

    """

    def __init__(self, n_trees=1, n_terminals=1, n_variables=1, n_iterations=10,
                 min_depth=1, max_depth=3, functions=[],
                 lower_bound=[0], upper_bound=[1]):
        """Initialization method.

        Args:
            n_trees (int): Number of trees.
            n_terminals (int): Number of terminal nodes.
            n_variables (int): Number of decision variables.
            n_iterations (int): Number of iterations.
            min_depth (int): Minimum depth of the trees.
            max_depth (int): Maximum depth of the trees.
            functions (list): Functions nodes.
            lower_bound (list): Lower bound list with the minimum possible values.
            upper_bound (list): Upper bound list with the maximum possible values.

        """

        logger.info('Overriding class: Space -> TreeSpace.')

        # Number of trees
        self.n_trees = n_trees

        # Override its parent class with the receiving arguments
        super(TreeSpace, self).__init__(n_trees, n_variables, n_iterations=n_iterations)

        # Number of terminal nodes
        self.n_terminals = n_terminals

        # Minimum depth of the trees
        self.min_depth = min_depth

        # Maximum depth of the trees
        self.max_depth = max_depth

        # List of functions nodes
        self.functions = functions

        # Now, we need to build this class up
        self._build(lower_bound, upper_bound)

        # Initializing the agents (structures that will hold trees' position and fitness)
        self._initialize_agents()

        # Creating the terminal nodes
        self._create_terminals()

        # Creating the trees
        self._create_trees()

        logger.info('Class overrided.')

    @property
    def n_trees(self):
        """int: Number of trees.

        """

        return self._n_trees

    @n_trees.setter
    def n_trees(self, n_trees):
        if not isinstance(n_trees, int):
            raise e.TypeError('`n_trees` should be an integer')
        if n_trees <= 0:
            raise e.ValueError('`n_trees` should be > 0')

        self._n_trees = n_trees

    @property
    def n_terminals(self):
        """int: Number of terminal nodes.

        """

        return self._n_terminals

    @n_terminals.setter
    def n_terminals(self, n_terminals):
        if not isinstance(n_terminals, int):
            raise e.TypeError('`n_terminals` should be an integer')
        if n_terminals <= 0:
            raise e.ValueError('`n_terminals` should be > 0')

        self._n_terminals = n_terminals

    @property
    def min_depth(self):
        """int: Minimum depth of the trees.

        """

        return self._min_depth

    @min_depth.setter
    def min_depth(self, min_depth):
        if not isinstance(min_depth, int):
            raise e.TypeError('`min_depth` should be an integer')
        if min_depth <= 0:
            raise e.ValueError('`min_depth` should be > 0')

        self._min_depth = min_depth

    @property
    def max_depth(self):
        """int: Maximum depth of the trees.

        """

        return self._max_depth

    @max_depth.setter
    def max_depth(self, max_depth):
        if not isinstance(max_depth, int):
            raise e.TypeError('`max_depth` should be an integer')
        if max_depth < self.min_depth:
            raise e.ValueError('`max_depth` should be >= `min_depth`')

        self._max_depth = max_depth

    @property
    def functions(self):
        """list: Functions nodes.

        """

        return self._functions

    @functions.setter
    def functions(self, functions):
        if not isinstance(functions, list):
            raise e.TypeError('`functions` should be a list')

        self._functions = functions

    @property
    def terminals(self):
        """list: Terminals nodes.

        """

        return self._terminals

    @terminals.setter
    def terminals(self, terminals):
        if not isinstance(terminals, list):
            raise e.TypeError('`terminals` should be a list')

        self._terminals = terminals

    @property
    def trees(self):
        """list: Trees instances (derived from the Node class).

        """

        return self._trees

    @trees.setter
    def trees(self, trees):
        if not isinstance(trees, list):
            raise e.TypeError('`trees` should be a list')

        self._trees = trees

    @property
    def best_tree(self):
        """Node: A best tree object from Node class.

        """

        return self._best_tree

    @best_tree.setter
    def best_tree(self, best_tree):
        if not isinstance(best_tree, Node):
            raise e.TypeError('`best_tree` should be a Node')

        self._best_tree = best_tree

    def _create_terminals(self):
        """Creates a list of terminals based on the Agent class.

        Returns:
            A list of terminals.

        """

        logger.debug('Running private method: create_terminals().')

        # Creating a list of terminals, which will be Agent instances
        self.terminals = [Agent(self.n_variables, self.n_dimensions) for _ in range(self.n_terminals)]

        logger.debug('Terminals created.')

    def _create_trees(self, algorithm='GROW'):
        """Creates a list of random trees using a specific algorithm.

        Args:
            algorithm (str): Algorithm's used to create the initial trees.

        Returns:
            The created trees and their fitness values.

        """

        logger.debug('Running private method: create_trees().')

        # Checks if the chosen algorithm is GROW
        if algorithm == 'GROW':
            # Creates a list of random trees
            self.trees = [self.grow(self.min_depth, self.max_depth) for _ in range(self.n_trees)]

        # Applies the first tree as the best one
        self.best_tree = copy.deepcopy(self.trees[0])

        logger.debug(
            f'Trees: {self.n_trees} | Depth: [{self.min_depth}, {self.max_depth}] | ' 
            f'Terminals: {self.n_terminals} | Functions: {self.functions} | Algorithm: {algorithm}.')

    def _initialize_agents(self):
        """Initialize agents' position array with uniform random numbers.

        """

        logger.debug('Running private method: initialize_agents().')

        # Iterates through all agents
        for agent in self.agents:
            # Iterates through all decision variables
            for j, (lb, ub) in enumerate(zip(self.lb, self.ub)):
                # For each decision variable, we generate uniform random numbers
                agent.position[j] = r.generate_uniform_random_number(lb, ub, agent.n_dimensions)

                # Applies the lower bound the agent's lower bound
                agent.lb[j] = lb

                # And also the upper bound
                agent.ub[j] = ub

        logger.debug('Agents initialized.')

    def _initialize_terminals(self):
        """Initialize terminals' position array with uniform random numbers.

        """

        # Iterates through all terminals
        for terminal in self.terminals:
            # Iterates through all decision variables
            for j, (lb, ub) in enumerate(zip(self.lb, self.ub)):
                # For each decision variable, we generate uniform random numbers
                terminal.position[j] = r.generate_uniform_random_number(lb, ub, terminal.n_dimensions)

                # Applies the lower bound the terminal's lower bound
                terminal.lb[j] = lb

                # And also the upper bound
                terminal.ub[j] = ub

    def grow(self, min_depth=1, max_depth=3):
        """It creates a random tree based on the GROW algorithm.

        References:
            S. Lukw. Two Fast Tree-Creation Algorithms for Genetic Programming.
            IEEE Transactions on Evolutionary Computation (2000).

        Args:
            min_depth (int): Minimum depth of the tree.
            max_depth (int): Maximum depth of the tree.

        Returns:
            A random tree based on the GROW algorithm.

        """

        # Re-initialize the terminals to provide diversity
        self._initialize_terminals()

        # If minimum depth equals the maximum depth
        if min_depth == max_depth:
            # Generates a terminal identifier
            terminal_id = int(r.generate_uniform_random_number(0, self.n_terminals))

            # Return the terminal node with its id and corresponding position
            return Node(terminal_id, 'TERMINAL', self.terminals[terminal_id].position)

        # If minimum depth is not equal to the maximum depth
        else:
            # Generates a node identifier
            node_id = int(r.generate_uniform_random_number(0, len(self.functions) + self.n_terminals))

            # If the identifier is a terminal
            if node_id >= len(self.functions):
                # Gathers its real identifier
                terminal_id = node_id - len(self.functions)

                # Return the terminal node with its id and corresponding position
                return Node(terminal_id, 'TERMINAL', self.terminals[terminal_id].position)

            # If the identifier is a function
            else:
                # Generates a new function node
                function_node = Node(self.functions[node_id], 'FUNCTION')

                # For every possible function argument
                for i in range(c.N_ARGS_FUNCTION[self.functions[node_id]]):
                    # Calls recursively the grow function and creates a temporary node
                    node = self.grow(min_depth + 1, max_depth)

                    # If it is not the root
                    if not i:
                        # The left child receives the temporary node
                        function_node.left = node

                    # If it is the first node
                    else:
                        # The right child receives the temporary node
                        function_node.right = node

                        # Flag to identify whether the node is a left child
                        node.flag = False

                    # The parent of the temporary node is the function node
                    node.parent = function_node

                return function_node
