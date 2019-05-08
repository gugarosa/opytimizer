import numpy as np

from opytimizer import Opytimizer
from opytimizer.core.function import Function
from opytimizer.math import benchmark
from opytimizer.optimizers.pso import PSO
from opytimizer.spaces.search import SearchSpace

# Number of agents
n_agents = 20

# Number of decision variables
n_variables = 2

# Number of running iterations
n_iterations = 1000

# Lower and upper bounds (has to be the same size as n_variables)
lower_bound = [-10, -10]
upper_bound = [10, 10]

# Creating the SearchSpace class
s = SearchSpace(n_agents=n_agents, n_iterations=n_iterations,
                n_variables=n_variables, lower_bound=lower_bound,
                upper_bound=upper_bound)

# Hyperparameters for the optimizer
hyperparams = {
    'w': 0.7,
    'c1': 1.7,
    'c2': 1.7
}

# Creating PSO's optimizer
p = PSO(hyperparams=hyperparams)

# Creating Function's object
f = Function(pointer=benchmark.sphere)

# Finally, we can create an Opytimizer class
o = Opytimizer(space=s, optimizer=p, function=f)

# Running the optimization task
o.start(history=True)
