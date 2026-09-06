import numpy as np

from opytimizer import Opytimizer
from opytimizer.optimizers.swarm import PSO
from opytimizer.spaces import SearchSpace
from opytimizer.utils.callback import CheckpointCallback


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

# Creates the space and optimizer
space = SearchSpace(n_agents, n_variables, lower_bound, upper_bound)
optimizer = PSO()

# Bundles every piece into Opytimizer class
opt = Opytimizer(space, optimizer, sphere, save_agents=False)

# Runs the optimization task
opt.start(n_iterations=10, callbacks=[CheckpointCallback(frequency=10)])

# Deletes the optimization object
del opt

# Loads the task from file and resumes it
# This deterministic objective and standard PSO can continue the same search.
# Other algorithms may restart iteration-local schedules when start() is called.
opt = Opytimizer.load("iter_10_checkpoint.pkl")
opt.start(n_iterations=25)
