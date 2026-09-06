# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Hypercomplex-based mathematical helpers.

"""

from collections.abc import Callable
from functools import wraps
from typing import TypeVar

import numpy as np
from numpy.typing import ArrayLike

_T = TypeVar("_T")


def span(
    array: np.ndarray,
    lower_bound: ArrayLike,
    upper_bound: ArrayLike,
) -> np.ndarray:
    """Spans a hypercomplex number between lower and upper bounds.

    Args:
        array: A 2-dimensional input array.
        lower_bound: Lower bounds to be spanned.
        upper_bound: Upper bounds to be spanned.

    Returns:
        Spanned values that can be used as decision variables.

    """

    lb = np.asarray(lower_bound)
    ub = np.asarray(upper_bound)

    array_span = (ub - lb) * (np.linalg.norm(array, axis=1) / np.sqrt(array.shape[1])) + lb

    return array_span


def span_to_hyper_value(
    lb: ArrayLike,
    ub: ArrayLike,
) -> Callable[[Callable[[np.ndarray], _T]], Callable[[np.ndarray], _T]]:
    """Decorate an objective to span its hypercomplex input between bounds.

    Args:
        lb: Lower bounds.
        ub: Upper bounds.

    Returns:
        Decorator preserving the objective's result while spanning its input.

    """

    def _span_to_hyper_value(f: Callable[[np.ndarray], _T]) -> Callable[[np.ndarray], _T]:
        @wraps(f)
        def __span_to_hyper_value(x: np.ndarray) -> _T:
            x = span(x, lb, ub)

            return f(x)

        return __span_to_hyper_value

    return _span_to_hyper_value
