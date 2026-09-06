# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

from opytimizer.functions.multi_objective import MultiObjectiveFunction


def test_function1(z: float) -> float:
    """Add the first objective's offset.

    Args:
        z: Decision-variable value.

    Returns:
        Value increased by two.

    """

    return z + 2


def test_function2(z: float) -> float:
    """Add the second objective's offset.

    Args:
        z: Decision-variable value.

    Returns:
        Value increased by five.

    """

    return z + 5


x = 0

h = MultiObjectiveFunction([test_function1, test_function2])

print(f"x: {x}")
print(f"f(x): {h.functions[0](x)}")
print(f"g(x): {h.functions[1](x)}")
print(f"h(x) = [f(x), g(x)]: {h(x)}")
