from opytimizer.functions import ConstrainedFunction


# Defines a function with a single input and a float-based return
def test_function(z):
    return z[0] + z[1]


# Defines a constraint where it returns a validity boolean
def c_1(z):
    return z[0] + z[1] < 0


# Declares `x`
x = [1, 1]

# Creates a ConstrainedFunction
f = ConstrainedFunction(test_function, [c_1], 10000.0)

# Prints out some properties
print(f'x: {x}')
print(f'f(x): {f(x)}')
