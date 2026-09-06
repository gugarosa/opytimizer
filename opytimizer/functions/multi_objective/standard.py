# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Standard multi-objective functions.

"""

from collections.abc import Callable

import numpy as np


class MultiObjectiveFunction:
    """A MultiObjectiveFunction class used to hold multi-objective functions.

    """

    def __init__(self, functions: list[Callable]) -> None:
        """Initialization method.

        Args:
            functions: Objective callables.

        Raises:
            TypeError: Functions are not supplied as a list containing only callables.

        Notes:
            Retain the callable list without copying. Evaluation passes the
            same position array to each objective and propagates their failures.

        """

        if not isinstance(functions, list):
            raise TypeError("`functions` should be a list.")
        if not all(callable(function) for function in functions):
            raise TypeError("`functions` should contain only callables.")

        self.functions = functions

    def __call__(self, x: np.ndarray) -> list[float]:
        """Calculates every objective value.

        Args:
            x: Array of positions.

        Returns:
            Objective values in the same order as the supplied callables.

        """

        return [function(x) for function in self.functions]
