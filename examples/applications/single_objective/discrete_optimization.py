import numpy as np

from opytimizer import Opytimizer
from opytimizer.optimizers.swarm import PSO
from opytimizer.spaces import SearchSpace
from opytimizer.utils.callback import DiscreteSearchCallback


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

# Defines the allowed values for performing the discrete search
allowed_values = [list(range(lb, ub, 2)) for lb, ub in zip(lower_bound, upper_bound)]

# Creates the space and optimizer
space = SearchSpace(n_agents, n_variables, lower_bound, upper_bound)
optimizer = PSO()

# Bundles every piece into Opytimizer class
opt = Opytimizer(space, optimizer, sphere, save_agents=False)

# Runs the optimization task
opt.start(
    n_iterations=5, callbacks=[DiscreteSearchCallback(allowed_values=allowed_values)]
)
