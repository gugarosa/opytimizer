from opytimizer.functions.multi_objective import MultiObjectiveWeightedFunction


# Defines some test functions
def test_function1(z):
    return z + 2


def test_function2(z):
    return z + 5


# Declares `x`
x = 0

# Any type of internal python-coded function
# can be used as a pointer
h = MultiObjectiveWeightedFunction([test_function1, test_function2], [0.5, 0.5])

# Testing out your new Function class
print(f'x: {x}')
print(f'f(x): {h.functions[0](x)}')
print(f'g(x): {h.functions[1](x)}')
print(f'h(x) = 0.5f(x) + 0.5g(x): {h(x)}')
