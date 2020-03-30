import numpy as np

import opytimizer.math.benchmark as b

# Creating an input vector
x = np.array([0.5, 0.5, 1, 1])
print(f'x: {x}')

# Calculating Sphere's function
y = b.sphere(x)
print(f'Sphere f(x): {y}')

# Calculating Exponential's function
y = b.exponential(x)
print(f'Exponential f(x): {y}')
