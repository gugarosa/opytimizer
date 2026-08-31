"""Constrained single-objective functions."""

from typing import Callable, List

import numpy as np


class ConstrainedFunction:
    """A ConstrainedFunction class used to hold constrained single-objective functions."""

    def __init__(
        self,
        function: Callable,
        constraints: List[Callable],
        penalty: float = 0.0,
    ) -> None:
        """Initialization method.

        Args:
            function: Callable that returns the fitness value.
            constraints: Constraints to be applied to the fitness function.
            penalty: Penalization factor when a constraint is not valid.

        """

        if not callable(function):
            raise TypeError("`function` should be callable")
        if not isinstance(constraints, list):
            raise TypeError("`constraints` should be a list")
        if not all(callable(constraint) for constraint in constraints):
            raise TypeError("every constraint should be callable")
        if not isinstance(penalty, (float, int)):
            raise TypeError("`penalty` should be a float or integer")
        if penalty < 0:
            raise ValueError("`penalty` should be >= 0")

        self.function = function
        self.constraints = constraints
        self.penalty = penalty

    def __call__(self, x: np.ndarray) -> float:
        """Calculates the constrained objective value.

        Args:
            x: Array of positions.

        Returns:
            (float): Constrained single-objective function fitness.

        """

        fitness = self.function(x)

        for constraint in self.constraints:
            if not constraint(x):
                fitness += self.penalty * fitness

        return fitness
