import numpy as np

import opytimizer.math.hyper as h

# Creates an array with ones
a = np.ones((2, 4))
print(f"Array: {a}")

# Declares lower and upper bounds
lb = np.array([-5, -5])
ub = np.array([-2, -2])

# Calculates the hypercomplex number norm
norm = h.norm(a)
print(f"Norm Array: {norm}")

# Spanning it into lower and upper bounds
span = h.span(a, lb, ub)
print(f"Spanned Array: {span}")
