"""Hypercomplex-based mathematical helpers.
"""

from functools import wraps

import numpy as np


def norm(array):
    """Calculates the norm over an array. It is used as the first step to map
    a hypercomplex number to a real-valued space.

    Args:
        array (np.array): A 2-dimensional input array.

    Returns:
        Norm calculated over the second axis, such as (2, 4) array shape
        will result in a norm (2, ) shape.

    """

    # Calculates the norm over a hypercomplex number
    array_norm = np.linalg.norm(array, axis=1)

    return array_norm


def span(array, lower_bound, upper_bound):
    """Spans a hypercomplex number between lower and upper bounds.

    Args:
        array (np.array): A 2-dimensional input array.
        lb (list, tuple, np.array): Lower bounds to be spanned.
        ub (list, tuple, np.array): Upper bounds to be spanned.

    Returns:
        Spanned values that can be used as decision variables.

    """

    # Forces lower and upper bounds to be arrays
    lb = np.asarray(lower_bound)
    ub = np.asarray(upper_bound)

    # Calculates the spanning function
    array_span = (ub - lb) * (norm(array) / np.sqrt(array.shape[1])) + lb

    return array_span


def span_to_hyper_value(lb, ub):
    """Spans a hyper-value between lower and upper bounds.

    Args:
        lb (list, tuple, np.array): Lower bounds.
        ub (list, tuple, np.array): Upper bounds.

    Returns:
        The output of the incoming objective function with a spanned input.

    """

    def _span_to_hyper_value(f):
        """Actually decorates the incoming objective function.

        Args:
            f (callable): Incoming objective function.

        Returns:
            The wrapped objective function.

        """

        @wraps(f)
        def __span_to_hyper_value(x):
            """Wraps the objective function for calculating its output.

            Args:
                x (np.array): Array of hyper-values.

            Returns:
                The objective function itself.

            """

            # Spans `x` between lower and upper bounds
            x = span(x, lb, ub)

            return f(x)

        return __span_to_hyper_value

    return _span_to_hyper_value
