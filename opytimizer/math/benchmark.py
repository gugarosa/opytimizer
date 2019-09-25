import numpy as np


def alpine1(x):
    """Alpine 1's function.

    It can be used with 'n' variables and has minimum at 0.
    Also, it is expected to be within [-10, 10] bounds.

    Args:
        x (np.array): An n-dimensional input array.

    Returns:
        y = sum(fabs(x * sin(x) + 0.1 * x))

    """

    # Declaring Alpine 1's function
    y = np.fabs(x * np.sin(x) + 0.1 * x)

    return np.sum(y)


def alpine2(x):
    """Alpine 2's function.

    It can be used with 'n' variables and has minimum at -2.808^n.
    Also, it is expected to be within [0, 10] bounds.

    Args:
        x (np.array): An n-dimensional input array.

    Returns:
        y = -prod(sqrt(x) * sin(x))

    """

    # Declaring Alpine 2's function
    y = np.sqrt(x) * np.sin(x)

    return -np.prod(y)


def chung_reynolds(x):
    """Chung Reynolds' function.

    It can be used with 'n' variables and has minimum at 0.
    Also, it is expected to be within [-100, 100] bounds.

    Args:
        x (np.array): An n-dimensional input array.

    Returns:
        y = (sum(x^2))^2

    """

    # Calculating Sphere's function
    s = sphere(x)

    return s ** 2


def csendes(x):
    """Csendes' function.

    It can be used with 'n' variables and has minimum at 0.
    Also, it is expected to be within [-1, 1] bounds.

    Args:
        x (np.array): An n-dimensional input array.

    Returns:
        y = sum(x^6 * (2 + sin(1 / x)))

    """

    # Declaring Csendes' function
    y = (x ** 6) * (2 + np.sin(1 / x))

    return np.sum(y)


# def deb1(x):
#     """Deb 1's function.

#     It can be used with 'n' variables and has minimum at 5^n.
#     Also, it is expected to be within [-1, 1] bounds.

#     Args:
#         x (np.array): An n-dimensional input array.

#     Returns:
#         y = (-1 / n) * sum(sin(5 * pi * x)^6)

#     """

#     return


# def deb3(x):
#     """Deb 3's function.

#     It can be used with 'n' variables and has minimum at 5^n.
#     Also, it is expected to be within [-1, 1] bounds.

#     Args:
#         x (np.array): An n-dimensional input array.

#     Returns:
#         y = (-1 / n) * sum(sin(5 * pi * (x^(3/4) - 0.05))^6)

#     """

#     return


def exponential(x):
    """Exponential's function.

    It can be used with 'n' variables and has minimum at -1.
    Also, it is expected to be within [-1, 1] bounds.

    Args:
        x (np.array): An n-dimensional input array.

    Returns:
        y = -exp(-0.5 * sum(x^2))

    """

    # Calculating Sphere's function
    s = sphere(x)

    return -np.exp(-0.5 * s)


# def griewank(x):
#     """Griewank's function.

#     It can be used with 'n' variables and has minimum at 0.
#     Also, it is expected to be within [-100, 100] bounds.

#     Args:
#         x (np.array): An n-dimensional input array.

#     Returns:
#         y = 1 + sum(x^2 / 4000) - prod(cos(x / sqrt(i)))

#     """

#     return


def rastringin(x):
    """Rastringin's function.

    It can be used with 'n' variables and has minimum at 0.
    Also, it is expected to be within [-5.12, 5.12] bounds.

    Args:
        x (np.array): An n-dimensional input array.

    Returns:
        y = A * n + sum(x^2 - A * cos(2 * pi * x))

    """

    # Declaring constant
    A = 10

    # Declaring Rastringin's function
    y = (x ** 2) - (A * np.cos(2 * np.pi * x))

    return A * x.shape[0] + np.sum(y)


def salomon(x):
    """Salomon's function.

    It can be used with 'n' variables and has minimum at 0.
    Also, it is expected to be within [-100, 100] bounds.

    Args:
        x (np.array): An n-dimensional input array.

    Returns:
        y = 1 - cos(2 * pi * sqrt(sum(x^2))) + 0.1 * sqrt(sum(x^2))

    """

    # Calculating Sphere's function
    s = sphere(x)

    return 1 - np.cos(2 * np.pi * np.sqrt(s)) + 0.1 * np.sqrt(s)


def schwefel(x):
    """Schwefel's function.

    It can be used with 'n' variables and has minimum at 0.
    Also, it is expected to be within [-500, 500] bounds.

    Args:
        x (np.array): An n-dimensional input array.

    Returns:
        y = 418.9829 * n - sum(x * sin(sqrt(fabs(x))))

    """

    # Declaring Schwefel's function
    y = x * np.sin(np.sqrt(np.fabs(x)))

    return 418.9829 * x.shape[0] - np.sum(y)


def sphere(x):
    """Sphere's function.

    It can be used with 'n' variables and has minimum at 0.
    Also, it is expected to be within [-5.12, 5.12] bounds.

    Args:
        x (np.array): An n-dimensional input array.

    Returns:
        y = sum(x^2)

    """

    # Declaring Sphere's function
    y = x ** 2

    return np.sum(y)
