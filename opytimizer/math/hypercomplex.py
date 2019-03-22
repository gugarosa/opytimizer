import numpy as np


def norm(array):
    """Calculates the norm over an array. It is used as the first step to map
    a hypercomplex number to a real-valued space.

    Args:
        array (np.array): A 2-dimensional input array.

    Returns:
        The norm calculated over the second axis, such as (2, 4) array shape
        will result in a norm (2, ) shape.

    """

    # Calculating the norm over the hypercomplex numbers
    array_norm = np.linalg.norm(array, axis=1)

    return array_norm


def span(array, lb, ub):
    """Spans a hypercomplex number between lower and upper bounds.

    Args:
        array (np.array): A 2-dimensional input array.
        lb (list | np.array): Lower bounds to be spanned.
        ub (list | np.array): Upper bounds to be spanned.

    Returns:
        A spanned value that can be used as decision variable in order to
        feed a fitness function.

    """

    # We need to force lower bound to be an array
    lb = np.array(lb)

    # Also the same thing goes for upper bound
    ub = np.array(ub)

    # Calculating span function
    array_span = (ub - lb) * (norm(array) / np.sqrt(array.shape[1])) + lb

    return array_span
