from opytimizer.core import Function


# Defines a function with a single input and a float-based return
def test_function(z):
    return z + 2


# Declares `x`
x = 0

# Any type of internal python-coded function
# can be used as a pointer
f = Function(test_function)

# Prints out some properties
print(f'x: {x}')
print(f'f(x): {f(x)}')
