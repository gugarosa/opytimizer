"""Standard multi-objective functions.
"""

import opytimizer.utils.exception as e
from opytimizer.core import Function
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class MultiObjectiveFunction:
    """A MultiObjectiveFunction class used to hold multi-objective functions."""

    def __init__(self, functions):
        """Initialization method.

        Args:
            functions (list): Pointers to functions that will return the fitness value.

        """

        logger.info("Creating class: MultiObjectiveFunction.")

        # List of functions
        self.functions = [Function(f) for f in functions] or []

        # Set built variable to 'True'
        self.built = True

        logger.debug(
            "Functions: %s | Built: %s", [f.name for f in self.functions], self.built
        )
        logger.info("Class created.")

    def __call__(self, x):
        """Callable to avoid using the `pointer` property.

        Args:
            x (np.array): Array of positions.

        Returns:
            Multi-objective function fitness.

        """

        # Defines a list to hold the total fitnesses
        z = []

        for f in self.functions:
            # Applies f(x)
            z.append(f.pointer(x))

        return z

    @property
    def functions(self):
        """list: Function's instances."""

        return self._functions

    @functions.setter
    def functions(self, functions):
        if not isinstance(functions, list):
            raise e.TypeError("`functions` should be a list")

        self._functions = functions
