import numpy as np

from opytimizer import Opytimizer
from opytimizer.optimizers.evolutionary import GP
from opytimizer.spaces import TreeSpace


def sphere(x):
    return np.sum(x**2)


# Random seed for experimental consistency
np.random.seed(0)

# Number of agents, terminals and decision variables
n_agents = 20
n_terminals = 2
n_variables = 2

# Minimum and maximum depths of the trees
min_depth = 2
max_depth = 5

# Functions nodes, lower and upper bounds
functions = ["SUM", "MUL", "DIV"]
lower_bound = [-10, -10]
upper_bound = [10, 10]

# Creates the space and optimizer
space = TreeSpace(
    n_agents,
    n_variables,
    lower_bound,
    upper_bound,
    n_terminals,
    min_depth,
    max_depth,
    functions,
)
optimizer = GP()

# Bundles every piece into Opytimizer class
opt = Opytimizer(space, optimizer, sphere, save_agents=False)

# Runs the optimization task
opt.start(n_iterations=1000)
