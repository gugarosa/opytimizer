"""Weighted multi-objective functions.
"""

import opytimizer.utils.exception as e
import opytimizer.utils.logging as l
from opytimizer.core import Function

logger = l.get_logger(__name__)


class WeightedFunction:
    """A WeightedFunction class used to hold weighted multi-objective functions.

    """

    def __init__(self, functions=None, weights=None):
        """Initialization method.

        Args:
            functions (list): Pointers to functions that will return the fitness value.
            weights (list): Weights for weighted-sum strategy.

        """

        logger.info('Creating class: WeightedFunction.')

        # List of functions
        if functions is None:
            self.functions = []
        else:
            self.functions = [Function(f) for f in functions]

        # List of weights
        if weights is None:
            self.weights = []
        else:
            self.weights = weights

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug('Functions: %s | Weights: %s | Built: %s',
                     [f.name for f in self.functions], self.weights, self.built)
        logger.info('Class created.')

    def __call__(self, x):
        """Callable to avoid using the `pointer` property.

        Args:
            x (np.array): Array of positions.

        Returns:
            Weighted multi-objective function fitness.

        """

        # Defines a variable to hold the total fitness
        z = 0

        # Iterates through every function
        for (f, w) in zip(self.functions, self.weights):
            # Applies w * f(x)
            z += w * f.pointer(x)

        return z

    @property
    def functions(self):
        """list: Function's instances.

        """

        return self._functions

    @functions.setter
    def functions(self, functions):
        if not isinstance(functions, list):
            raise e.TypeError('`functions` should be a list')

        self._functions = functions

    @property
    def weights(self):
        """list: Functions' weights.

        """

        return self._weights

    @weights.setter
    def weights(self, weights):
        if not isinstance(weights, list):
            raise e.TypeError('`weights` should be a list')
        if len(weights) != len(self.functions):
            raise e.SizeError('`weights` should have the same size of `functions`')

        self._weights = weights
