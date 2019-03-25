from opytimizer.core.function import Function
from opytimizer.functions.multi import Multi


# One should declare a function of x, where it should return a value
def test_function1(x):
    return x + 2

def test_function2(x):
    return x + 5


# Declaring x variable for further use
x = 0

# Functions can be used if your objective
# function is an internal python code
g = Multi(functions=[test_function1, test_function2])

# Testing out your new Function class
print(f'x: {x}')
print(f'f(x): {g.functions[0].pointer(x)}')
print(f'g(x): {g.functions[1].pointer(x)}')
print(f'f(g(x)): {g.pointer(x)}')
