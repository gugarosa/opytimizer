"""Constrained single-objective functions.
"""

import opytimizer.utils.exception as e
import opytimizer.utils.logging as l
from opytimizer.core import Function

logger = l.get_logger(__name__)


class ConstrainedFunction(Function):
    """A ConstrainedFunction class used to hold constrained single-objective functions.

    """

    def __init__(self, pointer, constraints, penalty=0.0):
        """Initialization method.

        Args:
            pointer (callable): Pointer to a function that will return the fitness value.
            constraints (list): Constraints to be applied to the fitness function.
            penalty (float): Penalization factor when a constraint is not valid.

        """

        logger.info('Overriding class: Function -> ConstrainedFunction.')

        # Overrides its parent class with the receiving arguments
        super(ConstrainedFunction, self).__init__(pointer)

        # List of constraints
        self.constraints = constraints or []

        # Penalization factor
        self.penalty = penalty

        # Logs the attributes
        logger.debug('Constraints: %s | Penalty: %s.',
                     self.constraints, self.penalty)
        logger.info('Class overrided.')

    @property
    def constraints(self):
        """list: Constraints to be applied to the fitness function.

        """

        return self._constraints

    @constraints.setter
    def constraints(self, constraints):
        if not isinstance(constraints, list):
            raise e.TypeError('`constraints` should be a list')

        self._constraints = constraints

    @property
    def penalty(self):
        """float: Penalization factor.

        """

        return self._penalty

    @penalty.setter
    def penalty(self, penalty):
        if not isinstance(penalty, (float, int)):
            raise e.TypeError('`penalty` should be a float or integer')
        if penalty < 0:
            raise e.ValueError('`penalty` should be >= 0')

        self._penalty = penalty

    def __call__(self, x):
        """Callable to avoid using the `pointer` property.

        Args:
            x (np.array): Array of positions.

        Returns:
            Constrained single-objective function fitness.

        """

        # Calculates the fitness function
        fitness = self.pointer(x)

        # For every possible constraint
        for constraint in self.constraints:
            # Checks if constraint is valid
            if constraint(x):
                pass

            # If the constraint is not valid
            else:
                # Penalizes the objective function
                fitness += self.penalty * fitness

        return fitness
