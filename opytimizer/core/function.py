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

        # Checking if pointer is actually a callable
        if hasattr(pointer, '__name__'):
            # If yes, applies the callable name
            self.name = pointer.__name__
        
        # If pointer comes from a class
        else:
            # Applies its name as the class' name
            self.name = pointer.__class__.__name__

        # Save the constraints for further inspection
        self.constraints = constraints

        # Also, we need to create a callable to point to the actual function
        self._create_pointer(pointer, constraints)

        # Indicates whether the function is built or not
        self.built = True

        logger.info('Class created.')
        logger.debug(f'Function: {self.name} | Constraints: {self.constraints} | Built: {self.built}')

    def __call__(self, x):
        """Defines a callable to this class in order to avoid using directly the property.

        Args:
            x (np.array): Array of positions to be calculated.

        Returns:
            The output of the objective function.

        """

        return self.pointer(x)

    @property
    def name(self):
        """str: Name of the function.

        """

        return self._name

    @name.setter
    def name(self, name):
        if not isinstance(name, str):
            raise e.TypeError('`name` should be a string')

        self._name = name

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

    def _create_pointer(self, pointer, constraints):
        """Wraps the fitness function if there are any constraints to be evaluated.

        Args:
            pointer (callable): Pointer to the actual function.
            constraints (list): Constraints to be applied.

        """

        # Checks if provided function has only one parameter
        if len(signature(pointer).parameters) > 1:
            # If not, raises an ArgumentError
            raise e.ArgumentError('`pointer` should only have 1 argument')

        def f_constrained(x):
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

        # Applying to the pointer property the return of constrained function
        self.pointer = f_constrained
