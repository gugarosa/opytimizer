from opytimizer.core.function import Function


# One should declare a function of x, where it should return a value
def test_function(x, y):
    return x + 2


# Declaring x variable for further use
x = 0
y = 0

# Functions can be used if your objective
# function is an internal python code
f = Function(pointer=test_function)

# Testing out your new Function class
print(f'\nx: {x}')
print(f'f(x): {f.pointer(x, y)}')
