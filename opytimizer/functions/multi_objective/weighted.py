# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Multi-objective weighted functions.

"""

from collections.abc import Callable

import numpy as np

from opytimizer.functions.multi_objective.standard import MultiObjectiveFunction


class MultiObjectiveWeightedFunction(MultiObjectiveFunction):
    """A MultiObjectiveWeightedFunction class used to hold multi-objective weighted functions.

    """

    def __init__(self, functions: list[Callable], weights: list[float]) -> None:
        """Initialization method.

        Args:
            functions: Objective callables.
            weights: Weights for weighted-sum strategy.

        Raises:
            TypeError: Objectives are not a list of callables or weights are not a list.
            ValueError: The number of weights does not match the number of objectives.

        Notes:
            Retain the callable and weight lists without copying. Weights are
            used as supplied rather than normalized.

        """

        super().__init__(functions)

        if not isinstance(weights, list):
            raise TypeError("`weights` should be a list.")
        if len(weights) != len(self.functions):
            raise ValueError("`weights` should match `functions`.")

        self.weights = weights

    def __call__(self, x: np.ndarray) -> float:
        """Calculates the weighted sum of all objective functions.

        Args:
            x: Array of positions.

        Returns:
            Sum of objective values multiplied by their corresponding weights.

        """

        return sum(weight * function(x) for function, weight in zip(self.functions, self.weights))
