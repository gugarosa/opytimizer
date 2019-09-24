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

        # Initially, an Agent needs its number of variables
        self.n_variables = n_variables

        # And also, its number of dimensions
        self.n_dimensions = n_dimensions

        # Create the position vector based on number of variables and dimensions
        self.position = np.zeros((n_variables, n_dimensions))

        # Fitness value is initialized with float's largest number
        self.fit = sys.float_info.max

        # Lower bounds are initialized as zero
        self.lb = np.zeros(n_variables)

        # Upper bounds are initialized as one
        self.ub = np.ones(n_variables)

    @property
    def n_variables(self):
        """int: Number of decision variables.

        """

        return self._n_variables

    @n_variables.setter
    def n_variables(self, n_variables):
        self._n_variables = n_variables

    @property
    def n_dimensions(self):
        """int: Dimension of search space.

        """

        return self._n_dimensions

    @n_dimensions.setter
    def n_dimensions(self, n_dimensions):
        self._n_dimensions = n_dimensions

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

    @property
    def lb(self):
        """np.array: Agent's lower bound value.

        """

        return self._lb

    @lb.setter
    def lb(self, lb):
        self._lb = lb

    @property
    def ub(self):
        """np.array: Agent's upper bound value.

        """

        return self._ub

    @ub.setter
    def ub(self, ub):
        self._ub = ub

    def check_limits(self):
        """Checks bounds limits of agent.

        """

        # Iterate through all decision variables
        for j, (lb, ub) in enumerate(zip(self.lb, self.ub)):
            # Clip the array based on variables' lower and upper bounds
            self.position[j] = np.clip(self.position[j], lb, ub)
