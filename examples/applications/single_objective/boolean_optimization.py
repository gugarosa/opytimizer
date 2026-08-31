import numpy as np

from opytimizer import Opytimizer
from opytimizer.optimizers.boolean import BPSO
from opytimizer.spaces import BooleanSpace

values = np.array([55, 10, 47, 5, 4])
weights = np.array([95, 4, 60, 32, 23])


def knapsack(x):
    selected = x.ravel()
    if weights @ selected > 100:
        return np.finfo(float).max
    return -(values @ selected)


# Random seed for experimental consistency
np.random.seed(0)

# Number of agents and decision variables
n_agents = 5
n_variables = 5

# Parameters for the optimizer
params = {
    "c1": np.random.randint(0, 2, size=(n_variables, 1)),
    "c2": np.random.randint(0, 2, size=(n_variables, 1)),
}

# Creates the space and optimizer
space = BooleanSpace(n_agents, n_variables)
optimizer = BPSO(params)

# Bundles every piece into Opytimizer class
opt = Opytimizer(space, optimizer, knapsack, save_agents=False)

# Runs the optimization task
opt.start(n_iterations=1000)
