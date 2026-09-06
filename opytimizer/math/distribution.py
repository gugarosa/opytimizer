# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Distribution-based mathematical generators.

"""

from math import gamma, pi, sin

import numpy as np


def generate_levy_distribution(beta: float = 0.1, size: int = 1) -> np.ndarray:
    """Generates a n-dimensional array based on a Lévy distribution.

    References:
        X.-S. Yang and S. Deb. Computers & Operations Research.
        Multiobjective Cuckoo Search for Design Optimization (2013).

    Args:
        beta: Stability exponent of the Lévy distribution.
        size: Size of array.

    Returns:
        Lévy distribution n-dimensional array.

    """

    num = gamma(1 + beta) * sin(pi * beta / 2)
    den = gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))

    sigma = (num / den) ** (1 / beta)

    u = np.random.normal(0, sigma, size=size)
    v = np.random.normal(size=size)

    levy_array = u / np.fabs(v) ** (1 / beta)

    return levy_array
