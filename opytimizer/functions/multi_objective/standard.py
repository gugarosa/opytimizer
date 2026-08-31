"""Standard multi-objective functions."""

from typing import Callable, List

import numpy as np


class MultiObjectiveFunction:
    """A MultiObjectiveFunction class used to hold multi-objective functions."""

    def __init__(self, functions: List[Callable]) -> None:
        """Initialization method.

        Args:
            functions: Objective callables.

        """

        if not isinstance(functions, list):
            raise TypeError("`functions` should be a list")
        if not all(callable(function) for function in functions):
            raise TypeError("every function should be callable")

        self.functions = functions

    def __call__(self, x: np.ndarray) -> List[float]:
        """Calculates every objective value.

        Args:
            x: Array of positions.

        Returns:
            (float): Multi-objective function fitness.

        """

        return [function(x) for function in self.functions]
