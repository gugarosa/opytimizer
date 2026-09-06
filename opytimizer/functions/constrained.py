# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Constrained single-objective functions.

"""

from collections.abc import Callable

import numpy as np


class ConstrainedFunction:
    """A ConstrainedFunction class used to hold constrained single-objective functions.

    """

    def __init__(
        self,
        function: Callable,
        constraints: list[Callable],
        penalty: float = 0.0,
    ) -> None:
        """Initialization method.

        Args:
            function: Callable that returns the fitness value.
            constraints: Constraints to be applied to the fitness function.
            penalty: Non-negative relative penalty applied for each invalid constraint.

        Raises:
            TypeError: The objective, constraint list, its callables, or the penalty have invalid types.
            ValueError: The penalty is negative.

        Notes:
            The function and constraint list are retained without copying.
            Each invalid constraint adds ``penalty * abs(fitness)`` to the
            current fitness, so penalties compound and zero fitness stays zero.

        """

        if not callable(function):
            raise TypeError("`function` should be callable.")
        if not isinstance(constraints, list):
            raise TypeError("`constraints` should be a list.")
        if not all(callable(constraint) for constraint in constraints):
            raise TypeError("`constraints` should contain only callables.")
        if not isinstance(penalty, (float, int)):
            raise TypeError("`penalty` should be a float or integer.")
        if penalty < 0:
            raise ValueError("`penalty` should be >= 0.")

        self.function = function
        self.constraints = constraints
        self.penalty = penalty

    def __call__(self, x: np.ndarray) -> float:
        """Calculates a minimized objective without rewarding constraint violations.

        Pass the same position array to the objective and every constraint.
        Failures from those callables propagate to the caller.

        Args:
            x: Array of positions.

        Returns:
            Penalized single-objective fitness.

        """

        fitness = self.function(x)

        for constraint in self.constraints:
            if not constraint(x):
                fitness += self.penalty * abs(fitness)

        return fitness
