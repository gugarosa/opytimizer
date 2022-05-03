"""3-D benchmarking functions plots.
"""

import matplotlib.pyplot as plt


def plot(points, title="", subtitle="", style="winter", colorbar=True):
    """Plots the surface from a 3-dimensional function.

    Args:
        points (np.array): Points to be plotted with shape equal to (3, n, n).
        title (str): Title of the plot.
        subtitle (str): Subtitle of the plot.
        style (str): Surface's style.
        colorbar (bool): If colorbar should be used or not.

    """

    # Creates the figure and axis
    fig = plt.figure(figsize=(9, 5))
    ax = plt.axes(projection="3d")

    # Defines some properties, such as labels, title, subtitle and ticks
    ax.set(xlabel="$x$", ylabel="$y$", zlabel="$f(x, y)$")
    ax.set_title(title, loc="left", fontsize=14)
    ax.set_title(subtitle, loc="right", fontsize=8, color="grey")
    ax.tick_params(labelsize=8)

    # Plots the wireframe and the surface
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

    # Displays the plot
    plt.show()
