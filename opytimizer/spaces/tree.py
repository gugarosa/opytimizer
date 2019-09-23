import numpy as np

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

    def __init__(self, n_agents=1, n_variables=2, n_iterations=10,
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
        super(TreeSpace, self).__init__(n_agents=n_agents, n_variables=n_variables,
                                        n_iterations=n_iterations, lower_bound=lower_bound, upper_bound=upper_bound)

        self.min_depth = min_depth

        self.max_depth = max_depth

        self.functions = functions

        self.terminals = terminals

        self.constants = None

        # Now, we need to build this class up
        self._build(lower_bound, upper_bound)

        # Initializing agents
        # self._initialize_agents()

        #
        self._check_constants()

        logger.debug(
            f'Depth: [{self.min_depth}, {self.max_depth}] | Functions: {self.functions} | Terminals: {self.terminals}.')

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
