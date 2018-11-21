import sys

import numpy as np
import opytimizer.utils.logging as l

logger = l.get_logger(__name__)


class Agent:
    """An agent class for all meta-heuristic optimization techniques.

    Properties:
        fit (float): Fitness value.
        n_dimensions (int): Dimension of search space.
        n_variables (int): Number of decision variables.
        position (np.array): [n_variables x n_dimensions] matrix of position values.

    """

    def __init__(self, n_variables=2, n_dimensions=1):
        """Initialization method.

        Args:
            n_dimensions (int): Dimension of search space.
            n_variables (int): Number of decision variables.

        """

        logger.info('Creating class: Agent')

        # Initially, an Agent needs its number of variables and dimensions
        self.n_variables = n_variables
        self.n_dimensions = n_dimensions

        # Create the position vector based on number of variables and dimensions
        self.position = np.zeros((self.n_variables, self.n_dimensions))

        # Fitness value is initialized with zero
        self.fit = sys.float_info.max

        # We will log some important information
        logger.debug(f'Size: ({self.n_variables}, {self.n_dimensions}) | Fitness: {self.fit}')
        logger.info('Class created.')
