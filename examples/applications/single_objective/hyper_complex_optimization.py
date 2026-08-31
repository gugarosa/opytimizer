import numpy as np

import opytimizer.math.hyper as h
from opytimizer import Opytimizer
from opytimizer.optimizers.swarm import PSO
from opytimizer.spaces import HyperComplexSpace

# Random seed for experimental consistency
np.random.seed(0)

# Number of agents, decision variables and dimensions
n_agents = 20
n_variables = 2
n_dimensions = 4

# Lower and upper bounds (has to be the same size as `n_variables`)
lower_bound = [-10, -10]
upper_bound = [10, 10]


# Wraps the objective function with a spanning decorator,
# allowing values to be spanned between lower and upper bounds
@h.span_to_hyper_value(lower_bound, upper_bound)
def sphere(x):
    return np.sum(x**2)


# Creates the space and optimizer
space = HyperComplexSpace(n_agents, n_variables, n_dimensions)
optimizer = PSO()

# Bundles every piece into Opytimizer class
opt = Opytimizer(space, optimizer, sphere, save_agents=False)

# Runs the optimization task
opt.start(n_iterations=1000)
