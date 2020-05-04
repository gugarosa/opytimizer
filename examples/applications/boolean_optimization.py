import numpy as np

import opytimizer.math.distribution as d
from opytimizer import Opytimizer
from opytimizer.core.function import Function
from opytimizer.optimizers.boolean.pso import bPSO
from opytimizer.spaces.boolean import BooleanSpace

# Random seed for experimental consistency
np.random.seed(0)

# Number of agents
n_agents = 10

# Number of decision variables
n_variables = 10

# Number of running iterations
n_iterations = 3

# Creating the BooleanSpace class
s = BooleanSpace(n_agents=n_agents, n_iterations=n_iterations, n_variables=n_variables)

# Hyperparameters for the optimizer
hyperparams = {
    'w': d.generate_bernoulli_distribution(0.5, size=n_variables),
    'c1': d.generate_bernoulli_distribution(0.5, size=n_variables),
    'c2': d.generate_bernoulli_distribution(0.5, size=n_variables)
}

# Creating bPSO's optimizer
p = bPSO(hyperparams=hyperparams)

# Creating Function's object
# f = Function(pointer=knapsack)

# Finally, we can create an Opytimizer class
# o = Opytimizer(space=s, optimizer=p, function=f)

# Running the optimization task
# history = o.start()
