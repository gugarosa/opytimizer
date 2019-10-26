import opytimizer.utils.exception as e
import opytimizer.utils.logging as l
from opytimizer.core.function import Function

logger = l.get_logger(__name__)


class WeightedFunction:
    """A WeightedFunction class for using with multi objective functions
    based on the weight sum strategy.

    """

    def __init__(self, functions=[], weights=[]):
        """Initialization method.

        Args:
            functions (list): Pointers to functions that will return the fitness value.
            weights (list): Weights for weighted sum strategy.

        """

        logger.info('Creating class: WeightedFunction.')

        # Creating a list to hold further Function's instances
        self.functions = functions

        # Creating a list of weights
        self.weights = weights

        # Now, we need to build this class up
        self._build(functions)

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

    def _build(self, functions):
        """This method serves as the object building process.

        One can define several commands here that does not necessarily
        needs to be on its initialization.

        Args:
            functions (list): Pointers to functions that will return the fitness value.

        """

        logger.debug('Running private method: build().')

        # Populating pointers with real functions
        self.functions = [Function(pointer=f) for f in functions]

        # Creating a multi-objective method strategy as the real pointer
        self.pointer = self._create_strategy()

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug(
            f'Functions: {[f.pointer.__name__ for f in self.functions]} | Weights: {self.weights} | Built: {self.built}')

    def _create_strategy(self):
        """Creates a multi-objective strategy as the real pointer.

        Returns:
            A callable based on the strategy.

        """

        def pointer(x):
            # Defining value to hold strategy
            z = 0

            # Iterate through every function
            for (f, w) in zip(self.functions, self.weights):
                # Apply w * f(x)
                z += w * f.pointer(x)

            return z

        return pointer
