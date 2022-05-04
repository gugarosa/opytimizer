"""Hypercomplex-based mathematical helpers.
"""

from functools import wraps
from typing import Any, List, Tuple, Union

import numpy as np


def norm(array: np.ndarray) -> np.ndarray:
    """Calculates the norm over an array. It is used as the first step to map
    a hypercomplex number to a real-valued space.

    Args:
        array: A 2-dimensional input array.

    Returns:
        (np.ndarray): Norm calculated over the second axis, such as (2, 4) array shape
        will result in a norm (2, ) shape.

    """

    array_norm = np.linalg.norm(array, axis=1)

    return array_norm


def span(
    array: np.ndarray,
    lower_bound: Union[List[Any], Tuple[Any, Any], np.ndarray],
    upper_bound: Union[List, Tuple, np.ndarray],
) -> np.ndarray:
    """Spans a hypercomplex number between lower and upper bounds.

    Args:
        array: A 2-dimensional input array.
        lb: Lower bounds to be spanned.
        ub: Upper bounds to be spanned.

    Returns:
        (np.ndarray): Spanned values that can be used as decision variables.

    """

    # Forces lower and upper bounds to be arrays
    lb = np.asarray(lower_bound)
    ub = np.asarray(upper_bound)

    # Calculates the spanning function
    array_span = (ub - lb) * (norm(array) / np.sqrt(array.shape[1])) + lb

    return array_span


def span_to_hyper_value(
    lb: Union[List[Any], Tuple[Any, Any], np.ndarray],
    ub: Union[List[Any], Tuple[Any, Any], np.ndarray],
) -> np.ndarray:
    """Spans a hyper-value between lower and upper bounds.

    Args:
        lb: Lower bounds.
        ub: Upper bounds.

    Returns:
        (np.ndarray): The output of the incoming objective function with a spanned input.

    """

    def _span_to_hyper_value(f: callable) -> callable:
        """Actually decorates the incoming objective function.

        Args:
            f: Incoming objective function.

        Returns:
            (callable): The wrapped objective function.

        """

        @wraps(f)
        def __span_to_hyper_value(x: np.ndarray) -> np.ndarray:
            """Wraps the objective function for calculating its output.

            Args:
                x: Array of hyper-values.

            Returns:
                (np.ndarray): The objective function itself.

            """

            # Spans `x` between lower and upper bounds
            x = span(x, lb, ub)

            return f(x)

        return __span_to_hyper_value

    return _span_to_hyper_value
