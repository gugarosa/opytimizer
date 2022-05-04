"""Standard multi-objective functions.
"""

from typing import List

import numpy as np

import opytimizer.utils.exception as e
from opytimizer.core import Function
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class MultiObjectiveFunction:
    """A MultiObjectiveFunction class used to hold multi-objective functions."""

    def __init__(self, functions: List[callable]) -> None:
        """Initialization method.

        Args:
            functions: Pointers to functions that will return the fitness value.

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

    def __call__(self, x: np.ndarray) -> float:
        """Callable to avoid using the `pointer` property.

        Args:
            x: Array of positions.

        Returns:
            (float): Multi-objective function fitness.

        """

        # Defines a list to hold the total fitnesses
        z = []

        for f in self.functions:
            # Applies f(x)
            z.append(f.pointer(x))

        return z

    @property
    def functions(self) -> List[callable]:
        """Function's instances."""

        return self._functions

    @functions.setter
    def functions(self, functions: List[callable]) -> None:
        if not isinstance(functions, list):
            raise e.TypeError("`functions` should be a list")

        self._functions = functions
