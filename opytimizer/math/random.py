"""Random-based mathematical generators.
"""

import numpy as np


def generate_binary_random_number(size=1):
    """Generates a binary random number or array based on an uniform distribution.

    Args:
        size (int): Size of array.

    Returns:
        An binary random number or array.

    """

    # Generates a random binary number or array
    binary_array = np.round(np.random.uniform(0, 1, size))

    return binary_array


def generate_gamma_random_number(shape=1.0, scale=1.0, size=1):
    """Generates an Erlang distribution based on gamma values.

    Args:
        shape (float): Shape parameter.
        scale (float): Scaling of the distribution.
        size (int): Size of array.

    Returns:
        An Erlang distribution array.

    """

    # Generates a random Erlang number or array
    gamma_array = np.random.gamma(shape, scale, size)

    return gamma_array


def generate_integer_random_number(low=0, high=1, exclude_value=None, size=None):
    """Generates a random number or array based on an integer distribution.

    Args:
        low (int): Lower interval.
        high (int): Higher interval.
        exclude_value (int): Value to be excluded from array.
        size (int): Size of array.

    Returns:
        An integer random number or array.

    """

    # Generates a random integer number or array
    integer_array = np.random.randint(low, high, size)

    # Checks if a value is supposed to be excluded
    if exclude_value is not None:
        # Creates a boolean array based on excluded value
        bool_array = (integer_array == exclude_value)

        # If the excluded value is present
        if np.any(bool_array):
            # Re-calls the function with same arguments
            return generate_integer_random_number(low, high, exclude_value, size)

    return integer_array


def generate_uniform_random_number(low=0.0, high=1.0, size=1):
    """Generates a random number or array based on an uniform distribution.

    Args:
        low (float): Lower interval.
        high (float): Higher interval.
        size (int): Size of array.

    Returns:
        An uniform random number or array.

    """

    # Generates a random uniform number or array
    uniform_array = np.random.uniform(low, high, size)

    return uniform_array


def generate_gaussian_random_number(mean=0.0, variance=1.0, size=1):
    """Generates a random number or array based on a gaussian distribution.

    Args:
        mean (float): Gaussian's mean value.
        variance (float): Gaussian's variance value.
        size (int): Size of array.

    Returns:
        A gaussian random number or array.

    """

    # Generates a random gaussian number or array
    gaussian_array = np.random.normal(mean, variance, size)

    return gaussian_array
