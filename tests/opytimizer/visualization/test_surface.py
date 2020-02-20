import numpy as np
import pytest

from opytimizer.visualization import surface


def test_convergence_plot():
    def f(x, y):
        return x**2 + y**2

    assert f(2, 2) == 8

    x = y = np.linspace(-10, 10, 10)

    x, y = np.meshgrid(x, y)

    z = f(x, y)

    points = np.asarray([x, y, z])

    surface.plot(points, colorbar=False)

    surface.plot(points, colorbar=True)
