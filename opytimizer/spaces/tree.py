"""Tree-based search space.
"""

import copy

import opytimizer.math.random as r
import opytimizer.utils.constant as c
import opytimizer.utils.exception as e
import opytimizer.utils.logging as l
from opytimizer.core import Agent, Node, Space

logger = l.get_logger(__name__)


class TreeSpace(Space):
    """A TreeSpace class for trees, agents, variables and methods
    related to a tree-based search space.

    """

    def __init__(self, n_agents, n_variables, lower_bound, upper_bound,
                 n_terminals=1, min_depth=1, max_depth=3, functions=None):
        """Initialization method.

        Args:
            n_agents (int): Number of agents (trees).
            n_variables (int): Number of decision variables.
            lower_bound (float, list, tuple, np.array): Minimum possible values.
            upper_bound (float, list, tuple, np.array): Maximum possible values.
            n_terminals (int): Number of terminal nodes.
            min_depth (int): Minimum depth of the trees.
            max_depth (int): Maximum depth of the trees.
            functions (list): Function nodes.

        """

        logger.info('Overriding class: Space -> TreeSpace.')

        # Defines missing override arguments
        n_dimensions = 1

        # Override its parent class with the receiving arguments
        super(TreeSpace, self).__init__(n_agents, n_variables, n_dimensions,
                                        lower_bound, upper_bound)

        # Number of terminal nodes
        self.n_terminals = n_terminals

        # Minimum depth of the trees
        self.min_depth = min_depth

        # Maximum depth of the trees
        self.max_depth = max_depth

        # Function nodes
        if functions is None:
            self.functions = []
        else:
            self.functions = functions

        # Creates terminals and trees
        self._create_terminals()
        self._create_trees()

        # Builds the class
        self.build()

        logger.info('Class overrided.')

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
        """list: Function nodes.

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
        """list: Trees (derived from the Node class).

        """

        return self._trees

    @trees.setter
    def trees(self, trees):
        if not isinstance(trees, list):
            raise e.TypeError('`trees` should be a list')

        self._trees = trees

    @property
    def best_tree(self):
        """Node: Best tree.

        """

        return self._best_tree

    @best_tree.setter
    def best_tree(self, best_tree):
        if not isinstance(best_tree, Node):
            raise e.TypeError('`best_tree` should be a Node')

        self._best_tree = best_tree

    def _create_terminals(self):
        """Creates a list of terminals.

        """

        # List of terminals
        self.terminals = [Agent(self.n_variables, self.n_dimensions,
                                self.lb, self.ub) for _ in range(self.n_terminals)]

    def _create_trees(self):
        """Creates a list of trees based on the GROW algorithm.

        """

        # List of trees
        self.trees = [self.grow(self.min_depth, self.max_depth)
                      for _ in range(self.n_agents)]

        # Defines a best tree
        self.best_tree = copy.deepcopy(self.trees[0])

        logger.debug('Depth: [%d, %d] | Terminals: %d | Function: %s.',
                     self.min_depth, self.max_depth, self.n_terminals, self.functions)

    def _initialize_agents(self):
        """Initializes agents with their positions and defines a best agent.

        """

        # Iterates through all agents
        for agent in self.agents:
            # Initializes the agent
            agent.fill_with_uniform()

        # Defines a best agent and a best tree
        self.best_agent = copy.deepcopy(self.agents[0])

    def _initialize_terminals(self):
        """Initializes terminals with their positions.

        """

        # Iterates through all terminals
        for terminal in self.terminals:
            # Initializes the terminal
            terminal.fill_with_uniform()

    def grow(self, min_depth=1, max_depth=3):
        """Creates a random tree based on the GROW algorithm.

        References:
            S. Luke. Two Fast Tree-Creation Algorithms for Genetic Programming.
            IEEE Transactions on Evolutionary Computation (2000).

        Args:
            min_depth (int): Minimum depth of the tree.
            max_depth (int): Maximum depth of the tree.

        Returns:
            Random tree based on the GROW algorithm.

        """

        # Re-initializes the terminals to provide diversity
        self._initialize_terminals()

        # If minimum depth equals the maximum depth
        if min_depth == max_depth:
            # Generates a terminal identifier
            terminal_id = r.generate_integer_random_number(0, self.n_terminals)

            return Node(terminal_id, 'TERMINAL', self.terminals[terminal_id].position)

        # Generates a node identifier
        node_id = r.generate_integer_random_number(
            0, len(self.functions) + self.n_terminals)

        # If the identifier is a terminal
        if node_id >= len(self.functions):
            # Gathers its real identifier
            terminal_id = node_id - len(self.functions)

            return Node(terminal_id, 'TERMINAL', self.terminals[terminal_id].position)

        # Generates a function node
        function_node = Node(self.functions[node_id], 'FUNCTION')

        # For every possible function argument
        for i in range(c.FUNCTION_N_ARGS[self.functions[node_id]]):
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
