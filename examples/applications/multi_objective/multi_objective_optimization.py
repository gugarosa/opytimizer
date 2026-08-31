import numpy as np

from opytimizer import Opytimizer
from opytimizer.functions import WeightedFunction
from opytimizer.optimizers.swarm import PSO
from opytimizer.spaces import SearchSpace


def rastrigin(x):
    return 10 * x.size + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def sphere(x):
    return np.sum(x**2)


# Random seed for experimental consistency
np.random.seed(0)

# Number of agents and decision variables
n_agents = 20
n_variables = 2

# Lower and upper bounds (has to be the same size as `n_variables`)
lower_bound = [-10, -10]
upper_bound = [10, 10]

# Creates the space, optimizer and function
space = SearchSpace(n_agents, n_variables, lower_bound, upper_bound)
optimizer = PSO()
function = WeightedFunction([rastrigin, sphere], [0.5, 0.5])

# Bundles every piece into Opytimizer class
opt = Opytimizer(space, optimizer, function, save_agents=False)

# Runs the optimization task
opt.start(n_iterations=1000)
