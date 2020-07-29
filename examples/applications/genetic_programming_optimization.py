import numpy as np
from opytimark.markers.n_dimensional import Sphere

from opytimizer import Opytimizer
from opytimizer.core.function import Function
from opytimizer.optimizers.evolutionary.gp import GP
from opytimizer.spaces.tree import TreeSpace

# Random seed for experimental consistency
np.random.seed(0)

# Firstly, we need to define the number of agents
n_trees = 10

# Also, each terminal will be rendered as an agent
n_terminals = 2

# Also, we need to define the number of decision variables
n_variables = 2

# We can also decide the number of iterations
n_iterations = 1000

# Minimum depth of the trees
min_depth = 2

# Maximum depth of the trees
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

# Hyperparameters for the optimizer
hyperparams = {
    'p_reproduction': 0.25,
    'p_mutation': 0.1,
    'p_crossover': 0.2,
    'prunning_ratio': 0.0
}

# Creating GP's optimizer
p = GP(hyperparams=hyperparams)

# Creating Function's object
f = Function(pointer=Sphere())

# Finally, we can create an Opytimizer class
o = Opytimizer(space=s, optimizer=p, function=f)

# Running the optimization task
history = o.start()
