from opytimizer.functions.internal import Internal

# One should declare a function of x
# and it should return a value
def test_function(x):
    return x + 2

# Declaring x variable for further use
x = 0

# Internal functions can be used if your objective
# function is an internal python code
f = Internal()

# Prior using the Internal class, you need to build it,
# passing the desired function as a parameter
f.build(test_function)

# Testing out your new Internal class
print('x value: ' + str(x))
print('f(x) value: ' + str(f.call(x)))