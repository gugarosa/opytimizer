"""Weighted-based multi-objective functions.
"""

import opytimizer.utils.exception as e
import opytimizer.utils.logging as l
from opytimizer.core.function import Function

logger = l.get_logger(__name__)


class WeightedFunction:
    """A WeightedFunction class for using with multi objective functions
    based on the weight sum strategy.

    """

    def __init__(self, functions=None, weights=None, constraints=None, penalty=0.0):
        """Initialization method.

        Args:
            functions (list): Pointers to functions that will return the fitness value.
            weights (list): Weights for weighted sum strategy.
            constraints (list): List of constraints to be applied to the fitness functions.
            penalty (float): Penalization factor when a constraint is not valid.

        """

        logger.info('Creating class: WeightedFunction.')
        
        # Checks if functions do not exist
        if functions is None:
            # Creates a list for compatibility
            self.functions = []
        
        # If functions really exist
        else:
            # Creating the functions property
            self.functions = functions

        # Checks if weights do not exist
        if weights is None:
            # Creates a list for compatibility
            self.weights = []

        # If weights really exist
        else:
            # Creating the weights property
            self.weights = weights

        # Now, we need to build this class up
        self._build(constraints, penalty)

        logger.info('Class created.')

    def __call__(self, x):
        """Defines a callable to this class in order to avoid using directly the property.

        Args:
            x (np.array): Array of positions to be calculated.

        Returns:
            The output of the objective function.

        """

        return self.pointer(x)

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

        self._weights = weights

    def _create_multi_objective(self):
        """Creates a multi-objective strategy as the real pointer.

        """

        def f_weighted(x):
            """Weights and sums the functions according to their weights.

            Args:
                x (np.array): Array to be evaluated.

            Returns:
                The value of the weighted function.

            """
            # Defining value to hold strategy
            z = 0

            # Iterates through every function
            for (f, w) in zip(self.functions, self.weights):
                # Applies w * f(x)
                z += w * f.pointer(x)

            return z

        # Applying to the pointer property the return of weighted method
        self.pointer = f_weighted

    def _build(self, constraints, penalty):
        """This method serves as the object building process.

        One can define several commands here that does not necessarily
        needs to be on its initialization.

        Args:
            constraints (list): List of constraints to be applied to the fitness function.
            penalty (float): Penalization factor when a constraint is not valid.

        """

        logger.debug('Running private method: build().')

        # Populating pointers with real functions
        self.functions = [Function(f, constraints, penalty) for f in self.functions]

        # Creating a multi-objective method strategy as the real pointer
        self._create_multi_objective()

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug('Functions: %s | Weights: %s | Built: %s',
                     [f.name for f in self.functions], self.weights, self.built)
