import matplotlib.pyplot as plt

import opytimizer.utils.exception as e


def plot(*args, labels=None, title='', subtitle='', grid=True, legend=True):
    """Plots the convergence graph of desired variables.

    Essentially, each variable is a list or numpy array
    with size equals to (iterations x 1).

    Args:
        labels (list): Labels to be applied for each plot in legend.
        title (str): The title of the plot.
        subtitle (str): The subtitle of the plot.
        grid (bool): If grid should be used or not.
        legend (bool): If legend should be displayed or not.

    """

    # Creating figure and axis subplots
    fig, ax = plt.subplots(figsize=(7, 5))

    # Defining some properties, such as axis labels
    ax.set(xlabel='iteration', ylabel='value')

    # Setting both title and subtitles
    ax.set_title(title, loc='left', fontsize=14)
    ax.set_title(subtitle, loc='right', fontsize=8, color='grey')

    # If grid usage is true
    if grid:
        # Adds the grid property to the axis
        ax.grid()

    # Check if labels argument exists
    if labels:
        # Also check if it is a list
        if not isinstance(labels, list):
            raise e.TypeError('`labels` should be a list')

        # And check if it has the same size of arguments
        if len(labels) != len(args):
            raise e.SizeError('`args` and `labels` should have the same size')

    # If labels argument does not exists
    else:
        # Creates a list with indicators
        labels = [f'variable_{i}' for i in range(len(args))]

    # Plotting the axis
    for (arg, label) in zip(args, labels):
        ax.plot(arg, label=label)

    # If legend usage is true
    if legend:
        # Adds the legend property to the axis
        ax.legend()

    # Displaying the plot
    plt.show()
