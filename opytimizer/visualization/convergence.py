"""Convergence plots.
"""

import matplotlib.pyplot as plt

import opytimizer.utils.exception as e


def plot(*args, labels=None, title='', subtitle='', xlabel='iteration', ylabel='value',
         grid=True, legend=True):
    """Plots the convergence graph of desired variables.

    Essentially, each variable is a list or numpy array
    with size equals to `n_iterations`.

    Args:
        labels (list): Labels to be applied for each plot in legend.
        title (str): Title of the plot.
        subtitle (str): Subtitle of the plot.
        xlabel (str): Axis `x` label.
        ylabel (str): Axis `y` label.
        grid (bool): If grid should be used or not.
        legend (bool): If legend should be displayed or not.

    """

    # Creates the figure and axis subplots
    _, ax = plt.subplots(figsize=(7, 5))

    # Defines some properties, such as labels, title and subtitle
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title, loc='left', fontsize=14)
    ax.set_title(subtitle, loc='right', fontsize=8, color='grey')

    # If grid usage is `True`
    if grid:
        # Adds the grid property to the axis
        ax.grid()

    # Checks if `labels` really exists
    if labels:
        # Checks a set of pre-defined `labels` conditions
        if not isinstance(labels, list):
            raise e.TypeError('`labels` should be a list')

        if len(labels) != len(args):
            raise e.SizeError('`args` and `labels` should have the same size')

    # If `labels` do not exists
    else:
        # Creates pre-defined `labels`
        labels = [f'variable_{i}' for i in range(len(args))]

    # Plots every argument
    for (arg, label) in zip(args, labels):
        ax.plot(arg, label=label)

    # If legend usage is `True`
    if legend:
        # Adds the legend property to the axis
        ax.legend()

    # Displays the plot
    plt.show()
