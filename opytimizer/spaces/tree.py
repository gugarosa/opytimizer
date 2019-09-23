import numpy as np
from anytree import Node

import opytimizer.math.random as r
import opytimizer.utils.logging as l
from opytimizer.core.agent import Agent
from opytimizer.core.space import Space

logger = l.get_logger(__name__)

#
N_CONSTANTS = 100


class TreeSpace(Space):
    """

    """

    def __init__(self, n_trees=1, n_variables=2, n_iterations=10,
                 min_depth=1, max_depth=1, functions=['SUB'], terminals=['PARAM', 'CONST'],
                 lower_bound=None, upper_bound=None):
        """Initialization method.

        Args:
            n_agents (int): Number of agents.
            n_variables (int): Number of decision variables.
            n_iterations (int): Number of iterations.
            min_depth (int):
            max_depth (int):
            functions (list):
            terminals (list):
            lower_bound (np.array): Lower bound array with the minimum possible values.
            upper_bound (np.array): Upper bound array with the maximum possible values.

        """

        logger.info('Overriding class: Space -> TreeSpace.')

        # Override its parent class with the receiving arguments
        super(TreeSpace, self).__init__(n_agents=len(terminals), n_variables=n_variables,
                                        n_iterations=n_iterations, lower_bound=lower_bound, upper_bound=upper_bound)

        self.n_trees = n_trees

        self.min_depth = min_depth

        self.max_depth = max_depth

        self.functions = functions

        self.terminals = terminals

        self.constants = None

        self.trees = []

        # Now, we need to build this class up
        self._build(lower_bound, upper_bound)

        # Initializing agents
        # self._initialize_agents()

        #
        self._check_constants()

        self._grow_trees()

        logger.debug(
            f'Trees: {self.n_trees} | Depth: [{self.min_depth}, {self.max_depth}] | Functions: {self.functions} | Terminals: {self.terminals}.')

        # We will log some important information
        logger.info('Class overrided.')

    def _check_constants(self):
        """

        """

        logger.debug('Running private method: check_constants().')

        #
        if 'CONST' in self.terminals:
            #
            self.constants = np.zeros((self.n_variables, N_CONSTANTS))

            #
            for j, (lb, ub) in enumerate(zip(self.lb, self.ub)):
                #
                self.constants[j] = r.generate_uniform_random_number(lb, ub, size=N_CONSTANTS)

            logger.debug('Constants initialized.')

    def _grow(self):
        """

        """

        if self.min_depth == self.max_depth:
            index = int(r.generate_uniform_random_number(0, len(self.terminals)))
            if self.terminals[index] == 'CONST':
                id = int(r.generate_uniform_random_number(0, N_CONSTANTS))
                return Node(self.terminals[index], id=id, status='CONSTANT')

            return Node(self.terminals[index], id=index, status='TERMINAL')

    def _grow_trees(self):

    
        for i in range(self.n_trees):
            self.trees.append(self._grow())
