import numpy as np

import opytimizer.visualization.surface as s


def f(x, y):
    return x**2 + y**2


# Defining both `x` and `y` arrays
x = y = np.linspace(-10, 10, 100)

# Creating a meshgrid from both `x` and `y`
x, y = np.meshgrid(x, y)

# Calculating f(x, y)
z = f(x, y)

# Creating final array for further plotting
points = np.asarray([x, y, z])

# Plotting the surface
s.plot(points, title='3-D Function Surface Plot',
       subtitle='Sphere: $f(x, y) = x^2 + y^2$',
       style='winter', colorbar=True)
