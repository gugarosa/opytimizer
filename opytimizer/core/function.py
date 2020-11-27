"""Standard objective function.
"""

from inspect import signature

import opytimizer.utils.exception as e
import opytimizer.utils.logging as l

logger = l.get_logger(__name__)


class Function:
    """A Function class for using with objective functions
    that will be further evaluated.

    It serves as the basis class for holding in-code related
    objective functions.

    """

    def __init__(self, pointer=callable, constraints=None, penalty=0.0):
        """Initialization method.

        Args:
            pointer (callable): This should be a pointer to a function that will return the fitness value.
            constraints (list): List of constraints to be applied to the fitness function.
            penalty (float): Penalization factor when a constraint is not valid.

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

        # Checks if constraints do not exist
        if constraints is None:
            # Creates an empty list for compatibility
            self.constraints = []

        # If constraints exist
        else:
            # Save the constraints for further inspection
            self.constraints = constraints

        # Creates a property for holding the penalization factor
        self.penalty = penalty

        # Also, we need to create a callable to point to the actual function
        self._create_pointer(pointer)

        # Indicates whether the function is built or not
        self.built = True

        logger.info('Class created.')
        logger.debug('Function: %s | Constraints: %s | Penalty: %s | Built: %s',
                     self.name, self.constraints, self.penalty, self.built)

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
    def penalty(self):
        """float: Constraint penalization factor.

        """

        return self._penalty

    @penalty.setter
    def penalty(self, penalty):
        if not isinstance(penalty, (float, int)):
            raise e.TypeError('`penalty` should be a float or integer')
        if penalty < 0:
            raise e.ValueError('`penalty` should be >= 0')

        self._penalty = penalty

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

    def _create_pointer(self, pointer):
        """Wraps the fitness function if there are any constraints to be evaluated.

        Args:
            pointer (callable): Pointer to the actual function.

        """

        # Checks if provided function has only one parameter
        if len(signature(pointer).parameters) > 1:
            # If not, raises an ArgumentError
            raise e.ArgumentError('`pointer` should only have 1 argument')

        def constrain_pointer(x):
            """Applies the constraints and penalizes the fitness function if one of them are not valid.

            Args:
                x (np.array): Array to be evaluated.

            Returns:
                The value of the fitness function.

            """

            # Calculates the fitness function
            fitness = pointer(x)

            # For every possible constraint
            for constraint in self.constraints:
                # Check if constraint is valid
                if constraint(x):
                    # If yes, just keep going
                    pass

                # If a constraint is not valid
                else:
                    # Penalizes the objective function
                    fitness += self.penalty * fitness

            # If all constraints are satisfied, return the fitness function
            return fitness

        # Applying to the pointer property the return of constrained function
        self.pointer = constrain_pointer
