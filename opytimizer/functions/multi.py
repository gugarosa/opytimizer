import opytimizer.utils.logging as l
from opytimizer.core.function import Function

logger = l.get_logger(__name__)


class Multi(Function):
    """A Multi Function class to hold multi objective functions
    that will be further evaluated.

    It will serve as the basis class for holding in-code related
    multi objective functions.

    """

    def __init__(self, functions=[], weights=[], method='weight_sum'):
        """Initialization method.

        Args:
            functions (list): This should be a list of pointers to functions
                that will return the fitness value.
            weights (list): List of weights for weighted sum strategy.
            method (str): Multi-objective function strategy method
                (weight_sum, ).

        """

        logger.info('Overriding class: Function -> Multi.')

        # Creating a list to hold further Function's instances
        self.functions = functions

        # Creating weights (when used with 'weight_sum' strategy).
        self.weights = weights

        # Creates an strategy method (weight_sum, )
        self.method = method

        # Now, we need to build this class up
        self._build(functions, method)

        logger.info('Class created.')

    @property
    def functions(self):
        """list: A list of Function's instances.

        """

        return self._functions

    @functions.setter
    def functions(self, functions):
        self._functions = functions

    @property
    def weights(self):
        """list: Weights (when used with 'weight_sum' strategy).

        """

        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights

    @property
    def method(self):
        """str: Strategy method (weight_sum, ).

        """

        return self._method

    @method.setter
    def method(self, method):
        self._method = method

    def _build(self, functions, method):
        """This method will serve as the object building process.

        One can define several commands here that does not necessarily
        needs to be on its initialization.

        Args:
            functions (list): This should be a list of pointers to functions
                that will return the fitness value.
            method (str): Strategy method.

        """

        logger.debug('Running private method: build().')

        # Populating pointers with real functions
        self.functions = self._create_functions(functions)

        # Creating a multi-objective method strategy as the real pointer
        self.pointer = self._create_strategy(method)

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug(
            f'Functions: {self.functions} | Built: {self.built}')

    def _create_functions(self, pointers):
        """Creates functions class instances using the provided list of pointers.

        Args:
            pointers (list): A list of pointers to create respective Function's instances.

        Returns:
            A list of Function's class instances.

        Raises:
            RuntimeError

        """

        # Creates an empty functions list
        functions = []

        # Checks if pointers is a list
        if type(pointers).__name__ == 'list':
            # Iterate through every item in list
            for pointer in pointers:
                # Creates a function class instance for each item
                functions.append(Function(pointer=pointer))

        # If not, raises an runtime error
        else:
            e = f"Property 'pointers' needs to be a list."
            logger.error(e)
            raise RuntimeError(e)

        return functions

    def _create_strategy(self, method):
        """Creates a multi-objective method strategy as the real pointer.

        Args:
            method (str): A string indicating what strategy method should be used.

        Returns:
            A callable based on defined strategy.

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
