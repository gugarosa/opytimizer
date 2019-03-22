import sys

import numpy as np
import opytimizer.utils.logging as l

logger = l.get_logger(__name__)


class Agent:
    """An agent class for all meta-heuristic optimization techniques.

    """

    def __init__(self, n_variables=2, n_dimensions=1):
        """Initialization method.

        Args:
            n_variables (int): Number of decision variables.
            n_dimensions (int): Dimension of search space.

        """

        logger.info('Creating class: Agent.')

        # Initially, an Agent needs its number of variables and dimensions
        self._n_variables = n_variables
        self._n_dimensions = n_dimensions

        # Create the position vector based on number of variables and dimensions
        self._position = np.zeros((n_variables, n_dimensions))

        # Fitness value is initialized with float's largest number
        self._fit = sys.float_info.max

        # We will log some important information
        logger.debug(
            f'Size: ({self._n_variables}, {self._n_dimensions}) | Fitness: {self._fit}.')

        logger.info('Class created.')

    @property
    def n_variables(self):
        """int: Number of decision variables.

        """

        return self._n_variables

    @property
    def n_dimensions(self):
        """int: Dimension of search space.

        """

        return self._n_dimensions

    @property
    def position(self):
        """np.array: A matrix of position values.

        """

        return self._position

    @position.setter
    def position(self, position):
        self._position = position

    @property
    def fit(self):
        """float: Agent's fitness value.

        """

        return self._fit

    @fit.setter
    def fit(self, fit):
        self._fit = fit
