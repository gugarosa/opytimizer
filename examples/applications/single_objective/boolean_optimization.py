import numpy as np
from opytimark.markers.boolean import Knapsack

import opytimizer.math.random as r
from opytimizer import Opytimizer
from opytimizer.core import Function
from opytimizer.optimizers.boolean import BPSO
from opytimizer.spaces import BooleanSpace

# Random seed for experimental consistency
np.random.seed(0)

# Number of agents and decision variables
n_agents = 5
n_variables = 5

# Parameters for the optimizer
params = {
    'c1': r.generate_binary_random_number(size=(n_variables, 1)),
    'c2': r.generate_binary_random_number(size=(n_variables, 1))
}

# Creates the space, optimizer and function
space = BooleanSpace(n_agents, n_variables)
optimizer = BPSO(params)
function = Function(Knapsack(values=(55, 10, 47, 5, 4), weights=(95, 4, 60, 32, 23), max_capacity=100))

# Bundles every piece into Opytimizer class
opt = Opytimizer(space, optimizer, function, save_agents=False)

# Runs the optimization task
opt.start(n_iterations=1000)
