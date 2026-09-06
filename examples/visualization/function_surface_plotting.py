# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

"""Plot a raw objective's surface with the optional Matplotlib dependency.

Install Matplotlib separately to run this recipe. It is not an Opytimizer runtime
dependency. Set ``MPLBACKEND=Agg`` when running without an interactive display.
The objective receives the same ``(n_variables, n_dimensions)`` position shape
used by a two-variable SearchSpace.

"""

import matplotlib.pyplot as plt
import numpy as np


def sphere(x: np.ndarray) -> float:
    """Evaluate the sphere objective.

    Args:
        x: Candidate position array.

    Returns:
        Sum of squared decision variables.

    """

    return np.sum(x**2)


coordinates = np.linspace(-5, 5, 50)
x, y = np.meshgrid(coordinates, coordinates)
points = np.column_stack((x.ravel(), y.ravel()))
z = np.array([sphere(point.reshape(2, 1)) for point in points]).reshape(x.shape)

figure = plt.figure()
axes = figure.add_subplot(projection="3d")
surface = axes.plot_surface(x, y, z, cmap="viridis")
axes.set(xlabel="x0", ylabel="x1", zlabel="Fitness", title="Sphere objective surface")
figure.colorbar(surface, ax=axes, shrink=0.6, label="Fitness")
figure.tight_layout()
plt.show()
