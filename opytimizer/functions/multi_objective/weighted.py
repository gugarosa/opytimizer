"""Multi-objective weighted functions.
"""

import opytimizer.utils.exception as e
import opytimizer.utils.logging as l
from opytimizer.functions.multi_objective.standard import MultiObjectiveFunction

logger = l.get_logger(__name__)


class MultiObjectiveWeightedFunction(MultiObjectiveFunction):
    """A MultiObjectiveWeightedFunction class used to hold multi-objective weighted functions.

    """

    def __init__(self, functions, weights):
        """Initialization method.

        Args:
            functions (list): Pointers to functions that will return the fitness value.
            weights (list): Weights for weighted-sum strategy.

        """

        logger.info('Overriding class: MultiObjectiveFunction -> MultiObjectiveWeightedFunction.')

        super(MultiObjectiveWeightedFunction, self).__init__(functions)

        # List of weights
        self.weights = weights or []

        logger.debug('Weights: %s', self.weights)
        logger.info('Class overrided.')

    def __call__(self, x):
        """Callable to avoid using the `pointer` property.

        Args:
            x (np.array): Array of positions.

        Returns:
            Multi-objective weighted function fitness.

        """

        # Defines a variable to hold the total fitness
        z = 0

        for (f, w) in zip(self.functions, self.weights):
            # Applies w * f(x)
            z += w * f.pointer(x)

        return z

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
