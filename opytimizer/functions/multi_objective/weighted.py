"""Multi-objective weighted functions."""

from typing import Callable, List

import numpy as np

from opytimizer.functions.multi_objective.standard import MultiObjectiveFunction


class MultiObjectiveWeightedFunction(MultiObjectiveFunction):
    """A MultiObjectiveWeightedFunction class used to hold multi-objective weighted functions."""

    def __init__(self, functions: List[Callable], weights: List[float]) -> None:
        """Initialization method.

        Args:
            functions: Objective callables.
            weights: Weights for weighted-sum strategy.

        """

        super().__init__(functions)

        if not isinstance(weights, list):
            raise TypeError("`weights` should be a list")
        if len(weights) != len(self.functions):
            raise ValueError("`weights` should match `functions`")

        self.weights = weights

    def __call__(self, x: np.ndarray) -> float:
        """Calculates the weighted sum of all objective functions.

        Args:
            x: Array of positions.

        """

        return sum(
            weight * function(x)
            for function, weight in zip(self.functions, self.weights)
        )
