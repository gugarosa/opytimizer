import numpy as np

from opytimizer import Opytimizer
from opytimizer.optimizers.misc import GS
from opytimizer.spaces import GridSpace


def sphere(x):
    return np.sum(x**2)


# Number of decision variables and step size of the grid
n_variables = 2
step = [0.1, 1]

# Lower and upper bounds (has to be the same size as `n_variables`)
lower_bound = [-10, -10]
upper_bound = [10, 10]

# Creates the space and optimizer
space = GridSpace(n_variables, step, lower_bound, upper_bound)
optimizer = GS()

# Bundles every piece into Opytimizer class
opt = Opytimizer(space, optimizer, sphere, save_agents=False)

# Runs the optimization task
opt.start()
