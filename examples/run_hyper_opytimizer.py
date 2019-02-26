import numpy as np

from opytimizer import Opytimizer
from opytimizer.functions.internal import Internal
from opytimizer.spaces.hyper import HyperSpace
from opytimizer.optimizers.pso import PSO


def sphere(x):
    # Declaring Sphere's function
    y = x ** 2

    return np.sum(y)


# Creating Internal's function
f = Internal(pointer=sphere)

# Number of agents
n_agents = 2

# Number of decision variables
n_variables = 2

# Number of space dimensions
n_dimensions = 4

# Number of running iterations
n_iterations = 10

# Lower and upper bounds (has to be the same size as n_variables)
lower_bound = [-10, -10]
upper_bound = [10, 10]

# Creating the HyperSpace class
s = HyperSpace(n_agents=n_agents, n_iterations=n_iterations,
                n_variables=n_variables, n_dimensions=n_dimensions,
                lower_bound=lower_bound, upper_bound=upper_bound)

# Hyperparameters for the optimizer
hyperparams = {
    'w': 0.7,
    'c1': 1.7,
    'c2': 1.7
}

# Creating PSO's optimizer
p = PSO(hyperparams=hyperparams)

# Finally, we can create an Opytimizer class
o = Opytimizer(space=s, optimizer=p, function=f)

# Running the optimization task
o.start()
