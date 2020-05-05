import numpy as np
from opytimark.markers.n_dimensional import Exponential, Sphere

from opytimizer import Opytimizer
from opytimizer.functions.weighted import WeightedFunction
from opytimizer.optimizers.swarm.fa import FA
from opytimizer.spaces.search import SearchSpace

# Random seed for experimental consistency
np.random.seed(0)

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
    'alpha': 0.5,
    'beta': 0.2,
    'gamma': 1.0
}

# Creating FA's optimizer
p = FA(hyperparams=hyperparams)

# Defining task's main function
z = WeightedFunction(functions=[Sphere(), Exponential()], weights=[0.5, 0.5])

# Finally, we can create an Opytimizer class
o = Opytimizer(space=s, optimizer=p, function=z)

# Running the optimization task
history = o.start()
