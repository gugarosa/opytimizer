import numpy as np
from opytimark.markers.n_dimensional import Sphere

from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.optimizers.swarm import PSO
from opytimizer.spaces import SearchSpace

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
function = Function(Sphere())

# Bundles every piece into Opytimizer class
opt = Opytimizer(space, optimizer, function, save_agents=False)

# Runs the optimization task
# Every call on `start` will the continue the optimization for `n_iterations`
# Note that the following lines achieves the same results as a 100-iteration running
opt.start(n_iterations=50)
opt.start(n_iterations=25)
opt.start(n_iterations=25)
