import numpy as np
import opytimizer.math.hypercomplex as h

# Creating an array with ones
a = np.ones((2, 4))
print(f'Array: {a}')

# Declaring lower bounds
lb = np.array([-5, -5])

# Also, we need to declare upper bounds
ub = np.array([-2, -2])

# Calculating the hypercomplex number norm
norm = h.norm(a)
print(f'Norm Array: {norm}')

# Spanning it into lower and upper bounds
span = h.span(a, lb, ub)
print(f'Spanned Array: {span}')
