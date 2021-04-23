import numpy as np
from opytimark.markers.n_dimensional import Sphere

from opytimizer import Opytimizer
from opytimizer.core.function import Function
from opytimizer.optimizers.evolutionary.gp import GP
from opytimizer.spaces.tree import TreeSpace

# Random seed for experimental consistency
np.random.seed(0)

# Number of trees, number of terminals, decision variables and iterations
n_trees = 10
n_terminals = 2
n_variables = 2
n_iterations = 1000

# Minimum and maximum depths of the trees
min_depth = 2
max_depth = 5

# List of functions nodes
functions = ['SUM', 'MUL', 'DIV']

# Finally, we define the lower and upper bounds
# Note that they have to be the same size as n_variables
lower_bound = (-10, -10)
upper_bound = (10, 10)

# Creating the TreeSpace object
s = TreeSpace(n_trees=n_trees, n_terminals=n_terminals, n_variables=n_variables,
              n_iterations=n_iterations, min_depth=min_depth, max_depth=max_depth,
              functions=functions, lower_bound=lower_bound, upper_bound=upper_bound)

# Parameters for the optimizer
params = {
    'p_reproduction': 0.25,
    'p_mutation': 0.1,
    'p_crossover': 0.2,
    'prunning_ratio': 0.0
}

# Creating GP's optimizer
p = GP(params=params)

# Creating Function's object
f = Function(pointer=Sphere())

# Finally, we can create an Opytimizer class
o = Opytimizer(space=s, optimizer=p, function=f)

# Running the optimization task
history = o.start()
