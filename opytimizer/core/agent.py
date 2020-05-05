import numpy as np

import opytimizer.utils.constants as c
import opytimizer.utils.exception as e
import opytimizer.utils.logging as l

logger = l.get_logger(__name__)


class Agent:
    """An Agent class for all optimization techniques.

    """

    def __init__(self, n_variables=1, n_dimensions=1):
        """Initialization method.

        Args:
            n_variables (int): Number of decision variables.
            n_dimensions (int): Number of dimensions.

        """

        # Initially, an agent needs its number of variables
        self.n_variables = n_variables

        # And also, its number of dimensions
        self.n_dimensions = n_dimensions

        # Create the position vector based on the number of variables and dimensions
        self.position = np.zeros((n_variables, n_dimensions))

        # Fitness value is initialized with float's largest number
        self.fit = c.FLOAT_MAX

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
        if not isinstance(n_variables, int):
            raise e.TypeError('`n_variables` should be an integer')
        if n_variables <= 0:
            raise e.ValueError('`n_variables` should be > 0')

        self._n_variables = n_variables

    @property
    def n_dimensions(self):
        """int: Number of dimensions.

        """

        return self._n_dimensions

    @n_dimensions.setter
    def n_dimensions(self, n_dimensions):
        if not isinstance(n_dimensions, int):
            raise e.TypeError('`n_dimensions` should be an integer')
        if n_dimensions <= 0:
            raise e.ValueError('`n_dimensions` should be > 0')

        self._n_dimensions = n_dimensions

    @property
    def position(self):
        """np.array: N-dimensional array of values.

        """

        return self._position

    @position.setter
    def position(self, position):
        if not isinstance(position, np.ndarray):
            raise e.TypeError('`position` should be a numpy array')

        self._position = position

    @property
    def fit(self):
        """float: Fitness value.

        """

        return self._fit

    @fit.setter
    def fit(self, fit):
        if not (isinstance(fit, float) or isinstance(fit, int)
                or isinstance(fit, np.int32) or isinstance(fit, np.int64)):
            raise e.TypeError('`fit` should be a float or integer')
        
        self._fit = fit

    @property
    def lb(self):
        """np.array: Lower bounds.

        """

        return self._lb

    @lb.setter
    def lb(self, lb):
        if not isinstance(lb, np.ndarray):
            raise e.TypeError('`lb` should be a numpy array')

        self._lb = lb

    @property
    def ub(self):
        """np.array: Upper bounds.

        """

        return self._ub

    @ub.setter
    def ub(self, ub):
        if not isinstance(ub, np.ndarray):
            raise e.TypeError('`ub` should be a numpy array')

        self._ub = ub

    def clip_limits(self):
        """Clips the agent's decision variables to the bounds limits.

        """

        # Iterates through all the decision variables
        for j, (lb, ub) in enumerate(zip(self.lb, self.ub)):
            # Clips the array based on variables' lower and upper bounds
            self.position[j] = np.clip(self.position[j], lb, ub)
