"""Random-based mathematical generators.
"""

import numpy as np


def generate_binary_random_number(size: int = 1) -> np.ndarray:
    """Generates a binary random number or array based on an uniform distribution.

    Args:
        size: Size of array.

    Returns:
        (np.ndarray): A binary random number or array.

    """

    binary_array = np.round(np.random.uniform(0, 1, size))

    return binary_array


def generate_exponential_random_number(scale: float = 1.0, size: int = 1) -> np.ndarray:
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
    shape: float = 1.0, scale: float = 1.0, size: int = 1
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
    low: int = 0,
    high: int = 1,
    exclude_value: int = None,
    size: int = None,
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

    if exclude_value is not None:
        bool_array = integer_array == exclude_value

        if np.any(bool_array):
            return generate_integer_random_number(low, high, exclude_value, size)

    return integer_array


def generate_uniform_random_number(
    low: float = 0.0, high: float = 1.0, size: int = 1
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
    mean: float = 0.0,
    variance: float = 1.0,
    size: int = 1,
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
