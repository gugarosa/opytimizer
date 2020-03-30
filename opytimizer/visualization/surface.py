import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

import opytimizer.utils.exception as e


def plot(points, title='', subtitle='', style='winter', colorbar=True):
    """Plots the surface from a 3-dimensional function.

    Args:
        points (np.array): Points to be plotted with shape equal to (3, n, n). 
        title (str): The title of the plot.
        subtitle (str): The subtitle of the plot.
        style (str): Surface's style.
        colorbar (bool): If colorbar should be used or not.

    """

    # Creating figure
    fig = plt.figure(figsize=(9, 5))

    # Creating the axis
    ax = plt.axes(projection='3d')

    # Defining some properties, such as axis labels
    ax.set(xlabel='$x$', ylabel='$y$', zlabel='$f(x, y)$')

    # Reducing the size of the ticks
    ax.tick_params(labelsize=8)

    # Setting both title and subtitles
    ax.set_title(title, loc='left', fontsize=14)
    ax.set_title(subtitle, loc='right', fontsize=8, color='grey')

    # PLotting the wireframe
    ax.plot_wireframe(points[0], points[1], points[2], color='grey')

    # Plotting the surface
    surface = ax.plot_surface(points[0], points[1], points[2],
                              rstride=1, cstride=1, cmap=style, edgecolor='none')

    # If colorbar usage is true
    if colorbar:
        # Adds the colorbar property to the figure
        fig.colorbar(surface, shrink=0.5, aspect=10)

    # Displaying the plot
    plt.show()
