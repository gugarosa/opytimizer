"""Convergence plots.
"""

from typing import List, Optional

import matplotlib.pyplot as plt

import opytimizer.utils.exception as e


def plot(
    *args,
    labels: Optional[List[str]] = None,
    title: Optional[str] = "",
    subtitle: Optional[str] = "",
    xlabel: Optional[str] = "iteration",
    ylabel: Optional[str] = "value",
    grid: Optional[bool] = True,
    legend: Optional[bool] = True,
) -> None:
    """Plots the convergence graph of desired variables.

    Essentially, each variable is a list or numpy array
    with size equals to `n_iterations`.

    Args:
        labels: Labels to be applied for each plot in legend.
        title: Title of the plot.
        subtitle: Subtitle of the plot.
        xlabel: Axis `x` label.
        ylabel: Axis `y` label.
        grid: If grid should be used or not.
        legend: If legend should be displayed or not.

    """

    # Creates the figure and axis subplots
    _, ax = plt.subplots(figsize=(7, 5))

    # Defines some properties, such as labels, title and subtitle
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title, loc="left", fontsize=14)
    ax.set_title(subtitle, loc="right", fontsize=8, color="grey")

    if grid:
        ax.grid()

    if labels:
        # Checks a set of pre-defined `labels` conditions
        if not isinstance(labels, list):
            raise e.TypeError("`labels` should be a list")

        if len(labels) != len(args):
            raise e.SizeError("`args` and `labels` should have the same size")

    else:
        # Creates pre-defined `labels`
        labels = [f"variable_{i}" for i in range(len(args))]

    # Plots every argument
    for (arg, label) in zip(args, labels):
        ax.plot(arg, label=label)

    if legend:
        ax.legend()

    # Displays the plot
    plt.show()
