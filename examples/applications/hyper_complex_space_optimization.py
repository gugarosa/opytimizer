import numpy as np
from opytimark.markers.n_dimensional import Sphere

import opytimizer.math.hyper as h
import opytimizer.utils.decorator as d
from opytimizer import Opytimizer
from opytimizer.core.function import Function
from opytimizer.optimizers.swarm.pso import PSO
from opytimizer.spaces.hyper_complex import HyperComplexSpace

# Random seed for experimental consistency
np.random.seed(0)

# Number of agents, decision variables, dimensions and iterations
n_agents = 20
n_variables = 2
n_dimensions = 4
n_iterations = 10000

# Lower and upper bounds (has to be the same size as n_variables)
lower_bound = (-10, -10)
upper_bound = (10, 10)

# Creating the HyperComplexSpace class
s = HyperComplexSpace(n_agents=n_agents, n_iterations=n_iterations,
                      n_variables=n_variables, n_dimensions=n_dimensions,
                      lower_bound=lower_bound, upper_bound=upper_bound)

# Wrapping the objective function with a spanning decorator
# This decorator allows values to be spanned between lower and upper bounds
@d.hyper_spanning(lower_bound, upper_bound)
def wrapper(x):
    z = Sphere()
    return z(x)

# Creating Function's object
f = Function(pointer=wrapper)

# Hyperparameters for the optimizer
params = {
    'w': 0.7,
    'c1': 1.7,
    'c2': 1.7
}

# Creating PSO's optimizer
p = PSO(params=params)

# Finally, we can create an Opytimizer class
o = Opytimizer(space=s, optimizer=p, function=f)

# Running the optimization task
history = o.start()
