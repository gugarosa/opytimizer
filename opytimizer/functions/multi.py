import opytimizer.utils.exception as e
import opytimizer.utils.logging as l
from opytimizer.core.function import Function

logger = l.get_logger(__name__)

# Constant to hold possible multi-objective strategies
METHODS = ['weight_sum']


class MultiFunction(Function):
    """A MultiFunction class for using with multi objective functions
    that will be further evaluated.

    It serves as the basis class for holding in-code related
    multi objective functions.

    """

    def __init__(self, functions=[], weights=[], method='weight_sum'):
        """Initialization method.

        Args:
            functions (list): Pointers to functions that will return the fitness value.
            weights (list): Weights for weighted sum strategy.
            method (str): Multi-objective function strategy method (e.g., weight_sum).

        """

        logger.info('Overriding class: Function -> Multi.')

        # Creating a list to hold further Function's instances
        self.functions = functions

        # Creating weights (when used with 'weight_sum' strategy).
        self.weights = weights

        # Creates an strategy method (e.g., weight_sum)
        self.method = method

        # Now, we need to build this class up
        self._build(functions, method)

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
        """list: Weights (when used with 'weight_sum' strategy).

        """

        return self._weights

    @weights.setter
    def weights(self, weights):
        if not isinstance(weights, list):
            raise e.TypeError('`weights` should be a list')

        self._weights = weights

    @property
    def method(self):
        """str: Strategy method (e.g., weight_sum).

        """

        return self._method

    @method.setter
    def method(self, method):
        if method not in METHODS:
            raise e.ArgumentError(f'`method` value should be in {METHODS}')

        self._method = method

    def _build(self, functions, method):
        """This method serves as the object building process.

        One can define several commands here that does not necessarily
        needs to be on its initialization.

        Args:
            functions (list): Pointers to functions that will return the fitness value.
            method (str): Strategy method.

        """

        logger.debug('Running private method: build().')

        # Populating pointers with real functions
        self.functions = [Function(pointer=f) for f in functions]

        # Creating a multi-objective method strategy as the real pointer
        self.pointer = self._create_strategy(method)

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug(
            f'Functions: {[f.pointer.__name__ for f in self.functions]} | Built: {self.built}')

    def _create_strategy(self, method):
        """Creates a multi-objective method strategy as the real pointer.

        Args:
            method (str): A string indicating what strategy method should be used.

        Returns:
            A callable based on the chosen strategy.

        """

        def pointer(x):
            # Check strategy method
            if method == 'weight_sum':
                # Defining value to hold strategy
                z = 0

                # Iterate through every function
                for (f, w) in zip(self.functions, self.weights):
                    # Apply w * f(x)
                    z += w * f.pointer(x)

                return z

        return pointer
