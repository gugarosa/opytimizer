# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Random mathematical helpers.

"""

import numpy as np


def integer(
    low: int = 0,
    high: int = 1,
    exclude: int | None = None,
    size: int | tuple[int, ...] | None = None,
) -> int | np.ndarray:
    """Return random integers from ``[low, high)`` without an excluded value.

    Args:
        low: Inclusive lower bound.
        high: Exclusive upper bound.
        exclude: Value to omit when it is within the requested interval.
        size: Output array shape, or ``None`` for a scalar.

    Returns:
        A random integer or an array of random integers with the requested shape.

    Raises:
        ValueError: The interval is empty or ``exclude`` removes its only possible value.

    """

    if exclude is None or not low <= exclude < high:
        return np.random.randint(low, high, size)

    if high - low == 1:
        raise ValueError("`exclude` cannot remove the only possible value.")

    values = np.random.randint(low, high - 1, size)

    return values + (values >= exclude)
