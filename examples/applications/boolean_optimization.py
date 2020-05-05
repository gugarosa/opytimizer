import numpy as np
from opytimark.markers.boolean import Knapsack

import opytimizer.math.random as r
from opytimizer import Opytimizer
from opytimizer.core.function import Function
from opytimizer.optimizers.boolean.bpso import BPSO
from opytimizer.spaces.boolean import BooleanSpace

# Random seed for experimental consistency
np.random.seed(0)

# Number of agents
n_agents = 5

# Number of decision variables
n_variables = 5

# Number of running iterations
n_iterations = 10

# Creating the BooleanSpace class
s = BooleanSpace(n_agents=n_agents, n_iterations=n_iterations, n_variables=n_variables)

# Hyperparameters for the optimizer
hyperparams = {
    'c1': r.generate_binary_random_number(size=(n_variables, 1)),
    'c2': r.generate_binary_random_number(size=(n_variables, 1))
}

# Creating BPSO's optimizer
p = BPSO(hyperparams=hyperparams)

# Creating Function's object
f = Function(pointer=Knapsack(values=[55, 10, 47, 5, 4], weights=[95, 4, 60, 32, 23], max_capacity=100))

# Finally, we can create an Opytimizer class
o = Opytimizer(space=s, optimizer=p, function=f)

# Running the optimization task
history = o.start()
