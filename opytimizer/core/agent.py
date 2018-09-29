import numpy as np
import opytimizer.utils.logging as l

logger = l.get_logger(__name__)


class Agent:
    """An agent class for all meta-heuristic optimization techniques.

    Properties:
        n_variables (int): Number of decision variables.
        n_dimensions (int): Dimension of search space.
        position (np.array): [n_variables x n_dimensions] matrix of position values.
        fit (float): Fitness value.

    """

    def __init__(self, n_variables=2, n_dimensions=1):
        """Initialization method.

        Args:
            n_variables (int): Number of decision variables.
            n_dimensions (int): Dimension of search space.

        """

        logger.info('Initializing Agent ...')

        # Initially, an Agent needs its number of variables and dimensions
        self.n_variables = n_variables
        self.n_dimensions = n_dimensions

        # Create the position vector based on number of variables and dimensions
        self.position = np.zeros((self.n_variables, self.n_dimensions))

        # Fitness value is initialized with zero
        self.fit = 0

        # We will log some important information
        logger.info('Agent created.')
        logger.info('Agent size: (' + str(self.n_variables) +
                    ',' + str(self.n_dimensions) + ')')
