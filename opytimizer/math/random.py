"""Random-based mathematical generators.
"""

from typing import Optional

import numpy as np


def generate_binary_random_number(size: Optional[int] = 1) -> np.ndarray:
    """Generates a binary random number or array based on an uniform distribution.

    Args:
        size: Size of array.

    Returns:
        (np.ndarray): A binary random number or array.

    """

    binary_array = np.round(np.random.uniform(0, 1, size))

    return binary_array


def generate_exponential_random_number(
    scale: Optional[float] = 1.0, size: Optional[int] = 1
) -> np.ndarray:
    """Generates a random number or array based on an exponential distribution.

    Args:
        scale: Scaling of the distribution.
        size: Size of array.

    Returns:
        (np.ndarray): An exponential random number or array.

    """

    exponential_array = np.random.exponential(scale, size)

    return exponential_array


def generate_gamma_random_number(
    shape: Optional[float] = 1.0, scale: Optional[float] = 1.0, size: Optional[int] = 1
) -> np.ndarray:
    """Generates an Erlang distribution based on gamma values.

    Args:
        shape: Shape parameter.
        scale: Scaling of the distribution.
        size: Size of array.

    Returns:
        (np.ndarray): An Erlang distribution array.

    """

    gamma_array = np.random.gamma(shape, scale, size)

    return gamma_array


def generate_integer_random_number(
    low: Optional[int] = 0,
    high: Optional[int] = 1,
    exclude_value: Optional[int] = None,
    size: Optional[int] = None,
) -> np.ndarray:
    """Generates a random number or array based on an integer distribution.

    Args:
        low: Lower interval.
        high: Higher interval.
        exclude_value: Value to be excluded from array.
        size: Size of array.

    Returns:
        (np.ndarray): An integer random number or array.

    """

    integer_array = np.random.randint(low, high, size)

    # Checks if a value is supposed to be excluded
    if exclude_value is not None:
        # Creates a boolean array based on excluded value
        bool_array = integer_array == exclude_value

        # If the excluded value is present
        if np.any(bool_array):
            # Re-calls the function with same arguments
            return generate_integer_random_number(low, high, exclude_value, size)

    return integer_array


def generate_uniform_random_number(
    low: Optional[float] = 0.0, high: Optional[float] = 1.0, size: Optional[int] = 1
) -> np.ndarray:
    """Generates a random number or array based on a uniform distribution.

    Args:
        low: Lower interval.
        high: Higher interval.
        size: Size of array.

    Returns:
        (np.ndarray): A uniform random number or array.

    """

    uniform_array = np.random.uniform(low, high, size)

    return uniform_array


def generate_gaussian_random_number(
    mean: Optional[float] = 0.0,
    variance: Optional[float] = 1.0,
    size: Optional[int] = 1,
) -> np.ndarray:
    """Generates a random number or array based on a gaussian distribution.

    Args:
        mean: Gaussian's mean value.
        variance: Gaussian's variance value.
        size: Size of array.

    Returns:
        (np.ndarray): A gaussian random number or array.

    """

    gaussian_array = np.random.normal(mean, variance, size)

    return gaussian_array
