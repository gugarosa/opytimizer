# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

from opytimizer.functions import ConstrainedFunction


def test_function(z: list[float]) -> float:
    """Add two decision variables.

    Args:
        z: Two decision-variable values.

    Returns:
        Sum of both values.

    """

    return z[0] + z[1]


def c_1(z: list[float]) -> bool:
    """Check whether the sum of two variables is negative.

    Args:
        z: Two decision-variable values.

    Returns:
        Whether the constraint is satisfied.

    """

    return z[0] + z[1] < 0


x = [1, 1]

f = ConstrainedFunction(test_function, [c_1], 10000.0)

print(f"x: {x}")
print(f"f(x): {f(x)}")
