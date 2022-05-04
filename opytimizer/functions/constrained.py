"""Constrained single-objective functions.
"""

from typing import List, Optional

import numpy as np

import opytimizer.utils.exception as e
from opytimizer.core import Function
from opytimizer.utils import logging

logger = logging.get_logger(__name__)


class ConstrainedFunction(Function):
    """A ConstrainedFunction class used to hold constrained single-objective functions."""

    def __init__(
        self,
        pointer: List[callable],
        constraints: List[callable],
        penalty: Optional[float] = 0.0,
    ) -> None:
        """Initialization method.

        Args:
            pointer: Pointer to a function that will return the fitness value.
            constraints: Constraints to be applied to the fitness function.
            penalty: Penalization factor when a constraint is not valid.

        """

        logger.info("Overriding class: Function -> ConstrainedFunction.")

        super(ConstrainedFunction, self).__init__(pointer)

        # List of constraints
        self.constraints = constraints or []

        # Penalization factor
        self.penalty = penalty

        logger.debug("Constraints: %s | Penalty: %s.", self.constraints, self.penalty)
        logger.info("Class overrided.")

    @property
    def constraints(self) -> List[callable]:
        """Constraints to be applied to the fitness function."""

        return self._constraints

    @constraints.setter
    def constraints(self, constraints: List[callable]) -> None:
        if not isinstance(constraints, list):
            raise e.TypeError("`constraints` should be a list")

        self._constraints = constraints

    @property
    def penalty(self) -> float:
        """Penalization factor."""

        return self._penalty

    @penalty.setter
    def penalty(self, penalty: float) -> None:
        if not isinstance(penalty, (float, int)):
            raise e.TypeError("`penalty` should be a float or integer")
        if penalty < 0:
            raise e.ValueError("`penalty` should be >= 0")

        self._penalty = penalty

    def __call__(self, x: np.ndarray) -> float:
        """Callable to avoid using the `pointer` property.

        Args:
            x: Array of positions.

        Returns:
            (float): Constrained single-objective function fitness.

        """

        # Calculates the fitness function
        fitness = self.pointer(x)

        for constraint in self.constraints:
            if constraint(x):
                pass

            else:
                # Penalizes the objective function
                fitness += self.penalty * fitness

        return fitness
