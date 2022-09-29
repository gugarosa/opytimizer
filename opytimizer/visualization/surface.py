"""3-D benchmarking functions plots.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot(
    points: np.ndarray,
    title: Optional[str] = "",
    subtitle: Optional[str] = "",
    style: Optional[str] = "winter",
    colorbar: Optional[bool] = True,
) -> None:
    """Plots the surface from a 3-dimensional function.

    Args:
        points: Points to be plotted with shape equal to (3, n, n).
        title: Title of the plot.
        subtitle: Subtitle of the plot.
        style: Surface's style.
        colorbar: If colorbar should be used or not.

    """

    fig = plt.figure(figsize=(9, 5))
    ax = plt.axes(projection="3d")

    ax.set(xlabel="$x$", ylabel="$y$", zlabel="$f(x, y)$")
    ax.set_title(title, loc="left", fontsize=14)
    ax.set_title(subtitle, loc="right", fontsize=8, color="grey")
    ax.tick_params(labelsize=8)

    ax.plot_wireframe(points[0], points[1], points[2], color="grey")
    surface = ax.plot_surface(
        points[0],
        points[1],
        points[2],
        rstride=1,
        cstride=1,
        cmap=style,
        edgecolor="none",
    )

    if colorbar:
        fig.colorbar(surface, shrink=0.5, aspect=10)

    plt.show()
