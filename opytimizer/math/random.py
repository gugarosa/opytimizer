"""Random mathematical helpers."""

import numpy as np


def integer(low=0, high=1, exclude=None, size=None):
    """Return random integers from ``[low, high)`` without an excluded value."""

    if exclude is None or not low <= exclude < high:
        return np.random.randint(low, high, size)

    values = np.random.randint(low, high - 1, size)

    return values + (values >= exclude)
