from inspect import signature

import opytimizer.utils.constants as c
import opytimizer.utils.exception as e
import opytimizer.utils.logging as l

logger = l.get_logger(__name__)


class Function:
    """A Function class for using with objective functions
    that will be further evaluated.

    It serves as the basis class for holding in-code related
    objective functions.

    """

    def __init__(self, pointer=callable, constraints=[]):
        """Initialization method.

        Args:
            pointer (callable): This should be a pointer to a function that will return the fitness value.
            constraints (list): List of constraints to be applied to the fitness function.

        """

        logger.info('Creating class: Function.')

        # Defining a function property just for further inspection
        self.function = pointer

        # Save the constraints for further inspection
        self.constraints = constraints

        # Also, we need a callable to point to the actual function
        self.pointer = self._wrapper(pointer, constraints)

        # Indicates whether the function is built or not
        self.built = True

        logger.info('Class created.')
        logger.debug(
            f'Fitness Function: {self.function.__name__} | Constraints: {self.constraints} | Built: {self.built}')

    @property
    def function(self):
        """callable: Fitness function to be used.

        """

        return self._function

    @function.setter
    def function(self, function):
        if not callable(function):
            raise e.TypeError('`function` should be a callable')
        if len(signature(function).parameters) > 1:
            raise e.ArgumentError('`function` should only have 1 argument')

        self._function = function

    @property
    def constraints(self):
        """list: List of constraints to be applied to the fitness function.

        """

        return self._constraints

    @constraints.setter
    def constraints(self, constraints):
        if not isinstance(constraints, list):
            raise e.TypeError('`constraints` should be a list')

        self._constraints = constraints

    @property
    def pointer(self):
        """callable: Points to the actual function.

        """

        return self._pointer

    @pointer.setter
    def pointer(self, pointer):
        if not callable(pointer):
            raise e.TypeError('`pointer` should be a callable')

        self._pointer = pointer

    @property
    def built(self):
        """bool: Indicate whether the function is built.

        """

        return self._built

    @built.setter
    def built(self, built):
        self._built = built

    def _wrapper(self, pointer, constraints):
        """Wraps the fitness function if there are any constraints to be evaluated.

        Args:
            pointer (callable): Pointer to the actual function.
            constraints (list): Constraints to be applied.

        Returns:
            The value of the fitness function.

        """

        def f(x):
            """Applies the constraints and penalizes the fitness function if one of them are not valid.

            Args:
                x (np.array): Array to be evaluated.

            Returns:
                The value of the fitness function.

            """

            # For every possible constraint
            for constraint in constraints:
                # Check if constraint is valid
                if constraint(x):
                    # If yes, just keep going
                    pass

                # If a single constraint is not valid
                else:
                    # Penalizes and returns the maximum possible value for the fitness function
                    return c.FLOAT_MAX

            # If all constraints are satisfied, return the fitness function
            return pointer(x)

        # Returns the function
        return f
