import numpy as np


def ackley1(x):
    """Ackley's First function.

    It can be used with 'n' variables and has minimum at 0.
    Also, it is expected to be within [-35, 35] bounds.

    Args:
        x (np.array): An n-dimensional input array.

    Returns:
        y = 20 - 20 * exp(-0.2 * sqrt(1 / n * sum(x^2))) + e - exp(1 / n * sum(cos(2 * pi * x)))

    """

    # Calculating the 1 / n term
    inv = 1 / x.shape[0]

    # Calculating first term
    term1 = -0.2 * np.sqrt(inv * np.sum(x ** 2))

    # Calculating second term
    term2 = inv * np.sum(np.cos(2 * np.pi * x))

    # Declaring Ackley's First function
    y = 20 - 20 * np.exp(term1) + np.e - np.exp(term2)

    return y


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


def brown(x):
    """Brown's function.

    It can be used with 'n' variables and has minimum at 0.
    Also, it is expected to be within [-1, 4] bounds.

    Args:
        x (np.array): An n-dimensional input array.

    Returns:
        y = sum((x_i^2)^(x_{i+1}^2+1) + (x_{i+1}^2)^(x_i^2+1))

    """

    # Calculating first term squares
    term1 = x[:-1] ** 2

    # Calculating second term squares
    term2 = x[1:] ** 2

    # Declaring Brown's function
    y = np.sum(term1 ** (term2 + 1) + term2 ** (term1 + 1))

    return y


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


def cosine_mixture(x):
    """Cosine's Mixture function.

    It can be used with 'n' variables and has minimum at 0.1 * n.
    Also, it is expected to be within [-1, 1] bounds.

    Args:
        x (np.array): An n-dimensional input array.

    Returns:
        y = 0.1 * sum(cos(5 * PI * x)) - sum(x^2)

    """

    # Calculating first term
    term1 = np.sum(np.cos(5 * np.pi * x))

    # Calculating second term
    term2 = np.sum(x ** 2)

    # Declaring Cosine's Mixture function
    y = 0.1 * term1 - term2

    return y


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


def deb1(x):
    """Deb 1's function.

    It can be used with 'n' variables and has minimum at -1.
    Also, it is expected to be within [-1, 1] bounds.

    Args:
        x (np.array): An n-dimensional input array.

    Returns:
        y = (-1 / n) * sum(sin(5 * pi * x)^6)

    """

    # Calculating partial term
    term = np.sum(np.sin(5 * np.pi * x) ** 6)

    # Declaring Deb 1's function
    y = -1 / x.shape[0] * term

    return y


def deb2(x):
    """Deb 2's function.

    It can be used with 'n' variables and has minimum at -1.
    Also, it is expected to be within [-1, 1] bounds.

    Args:
        x (np.array): An n-dimensional input array.

    Returns:
        y = (-1 / n) * sum(sin(5 * pi * (x^(3/4) - 0.05))^6)

    """

    # Calculating partial term
    term = np.sum(np.sin(5 * np.pi * (x ** (3/4) - 0.05)) ** 6)

    # Declaring Deb 1's function
    y = -1 / x.shape[0] * term

    return y


# def dixon_price(x):
#     """Dixon-Price's function.

#     It can be used with 'n' variables and has minimum at 0.
#     Also, it is expected to be within [-10, 10] bounds.

#     Args:
#         x (np.array): An n-dimensional input array.

#     Returns:
#         y =

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


# def levy(x):
#     """Lévy's function.

#     It can be used with 'n' variables and has minimum at 0.
#     Also, it is expected to be within [-10, 10] bounds.

#     Args:
#         x (np.array): An n-dimensional input array.

#     Returns:
#         y =

#     """

#     return


# def pathological(x):
#     """Pathological's function.

#     It can be used with 'n' variables and has minimum at 0.
#     Also, it is expected to be within [-100, 100] bounds.

#     Args:
#         x (np.array): An n-dimensional input array.

#     Returns:
#         y =

#     """

#     return


# def powell_sum(x):
#     """Powell Sum's function.

#     It can be used with 'n' variables and has minimum at 0.
#     Also, it is expected to be within [-1, 1] bounds.

#     Args:
#         x (np.array): An n-dimensional input array.

#     Returns:
#         y =

#     """

#     return


# def qing(x):
#     """Qing's function.

#     It can be used with 'n' variables and has minimum at 0.
#     Also, it is expected to be within [-500, 500] bounds.

#     Args:
#         x (np.array): An n-dimensional input array.

#     Returns:
#         y =

#     """

#     return


# def quartic(x):
#     """Quartic's function.

#     It can be used with 'n' variables and has minimum at 0.
#     Also, it is expected to be within [-1.28, 1.28] bounds.

#     Args:
#         x (np.array): An n-dimensional input array.

#     Returns:
#         y =

#     """

#     return


def quintic(x):
    """Quintic's function.

    It can be used with 'n' variables and has minimum at 0.
    Also, it is expected to be within [-10, 10] bounds.

    Args:
        x (np.array): An n-dimensional input array.

    Returns:
        y = sum(abs(x^5 - 3x^4 + 4x^3 + 2x^2 - 10x - 4))

    """

    # Declaring Quintic's function
    y = np.sum(np.fabs(x ** 5 - 3 * x ** 4 + 4 *
                       x ** 3 + 2 * x ** 2 - 10 * x - 4))

    return y


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


def schumer_steiglitz(x):
    """Schumer Steiglitz's function.

    It can be used with 'n' variables and has minimum at 0.
    Also, it is expected to be within [-100, 100] bounds.

    Args:
        x (np.array): An n-dimensional input array.

    Returns:
        y = sum(x^4)

    """

    # Declaring Schumer Steiglitz's function
    y = np.sum(x ** 4)

    return y


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


# def streched_v_sine_wave(x):
#     """Streched V Sine Wave's function.

#     It can be used with 'n' variables and has minimum at 0.
#     Also, it is expected to be within [0, 10] bounds.

#     Args:
#         x (np.array): An n-dimensional input array.

#     Returns:
#         y =

#     """

#     return


# def sum_different_powers(x):
#     """Sum of Different Powers' function.

#     It can be used with 'n' variables and has minimum at 0.
#     Also, it is expected to be within [-1, 1] bounds.

#     Args:
#         x (np.array): An n-dimensional input array.

#     Returns:
#         y =

#     """

#     return


# def sum_squares(x):
#     """Sum of Squares' function.

#     It can be used with 'n' variables and has minimum at 0.
#     Also, it is expected to be within [-10, 10] bounds.

#     Args:
#         x (np.array): An n-dimensional input array.

#     Returns:
#         y =

#     """

#     return


def styblinski_tang(x):
    """Styblinski-Tang's function.

    It can be used with 'n' variables and has minimum at -78.332.
    Also, it is expected to be within [-5, 5] bounds.

    Args:
        x (np.array): An n-dimensional input array.

    Returns:
        y = 1/2 * sum(x^4 - 16x^2 + 5x)

    """

    # Calculating partial term
    term = np.sum(x ** 4 - 16 * x ** 2 + 5 * x)

    # Declaring Styblinski-Tang's function
    y = 1 / 2 * term

    return y


# def trigonometric1(x):
#     """Trigonometric 1's function.

#     It can be used with 'n' variables and has minimum at 0.
#     Also, it is expected to be within [0, PI] bounds.

#     Args:
#         x (np.array): An n-dimensional input array.

#     Returns:
#         y =

#     """

#     return


# def trigonometric2(x):
#     """Trigonometric 2's function.

#     It can be used with 'n' variables and has minimum at 1.
#     Also, it is expected to be within [-500, 500] bounds.

#     Args:
#         x (np.array): An n-dimensional input array.

#     Returns:
#         y =

#     """

#     return


# def wavy(x):
#     """Wavy's function.

#     It can be used with 'n' variables and has minimum at 0.
#     Also, it is expected to be within [-PI, PI] bounds.

#     Args:
#         x (np.array): An n-dimensional input array.

#     Returns:
#         y = 1/n * sum(1 - cos(10x) * e^(0.5 * x^2))

#     """

#     # Calculating partial term
#     term =

#     # Declaring Wavy's function
#     y =

#     return y


# def xin_she_yang1(x):
#     """Xin-She Yang 1's function.

#     It can be used with 'n' variables and has minimum at 0.
#     Also, it is expected to be within [-5, 5] bounds.

#     Args:
#         x (np.array): An n-dimensional input array.

#     Returns:
#         y =

#     """

#     return


# def xin_she_yang2(x):
#     """Xin-She Yang 2's function.

#     It can be used with 'n' variables and has minimum at 0.
#     Also, it is expected to be within [-2PI, 2PI] bounds.

#     Args:
#         x (np.array): An n-dimensional input array.

#     Returns:
#         y =

#     """

#     return


# def xin_she_yang3(x):
#     """Xin-She Yang 3's function.

#     It can be used with 'n' variables and has minimum at -1.
#     Also, it is expected to be within [-20, 20] bounds.

#     Args:
#         x (np.array): An n-dimensional input array.

#     Returns:
#         y =

#     """

#     return


# def xin_she_yang4(x):
#     """Xin-She Yang 4's function.

#     It can be used with 'n' variables and has minimum at -1.
#     Also, it is expected to be within [-10, 10] bounds.

#     Args:
#         x (np.array): An n-dimensional input array.

#     Returns:
#         y =

#     """

#     return


# def zakharov(x):
#     """Zakharov's function.

#     It can be used with 'n' variables and has minimum at 0.
#     Also, it is expected to be within [-5, 10] bounds.

#     Args:
#         x (np.array): An n-dimensional input array.

#     Returns:
#         y =

#     """

#     return
