import opytimizer.utils.exception as e
import opytimizer.utils.logging as l
from opytimizer.core.function import Function

logger = l.get_logger(__name__)


class WeightedFunction:
    """A WeightedFunction class for using with multi objective functions
    based on the weight sum strategy.

    """

    def __init__(self, functions=[], weights=[], constraints=[]):
        """Initialization method.

        Args:
            functions (list): Pointers to functions that will return the fitness value.
            weights (list): Weights for weighted sum strategy.
            constraints (list): List of constraints to be applied to the fitness functions.

        """

        logger.info('Creating class: WeightedFunction.')

        # Creating a list to hold further Function's instances
        self.functions = functions

        # Creating a list of weights
        self.weights = weights

        # Now, we need to build this class up
        self._build(functions, constraints)

        logger.info('Class created.')

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

    def __call__(self, x):
        """Defines a callable to this class in order to avoid using directly the property.

        Args:
            x (np.array): Array of positions to be calculated.

        Returns:
            The output of the objective function.

        """

        return self.pointer(x)

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

    def _build(self, functions, constraints):
        """This method serves as the object building process.

        One can define several commands here that does not necessarily
        needs to be on its initialization.

        Args:
            functions (list): Pointers to functions that will return the fitness value.
            constraints (list): List of constraints to be applied to the fitness function.

        """

        logger.debug('Running private method: build().')

        # Populating pointers with real functions
        self.functions = [Function(f, constraints) for f in functions]

        # Creating a multi-objective method strategy as the real pointer
        self._create_multi_objective()

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug(f'Functions: {[f.name for f in self.functions]} | Weights: {self.weights} | Built: {self.built}')
