import numpy as np

from opytimizer import Opytimizer
from opytimizer.core.function import Function
from opytimizer.math import benchmark
from opytimizer.optimizers.gp import GP
from opytimizer.spaces.tree import TreeSpace

# Firstly, we need to define the number of agents
n_trees = 10

# Also, each terminal will be rendered as an agent
n_terminals = 2

# Also, we need to define the number of decision variables
n_variables = 2

# We can also decide the number of iterations
n_iterations = 10

# Minimum depth of the trees
min_depth = 2

# Maximum depth of the trees
max_depth = 5

# List of functions nodes
functions = ['SUM', 'MUL', 'DIV']

# Finally, we define the lower and upper bounds
# Note that they have to be the same size as n_variables
lower_bound = [-10, -10]
upper_bound = [10, 10]

# Creating the TreeSpace object
s = TreeSpace(n_trees=n_trees, n_terminals=n_terminals, n_variables=n_variables,
              n_iterations=n_iterations, min_depth=min_depth, max_depth=max_depth,
              functions=functions, lower_bound=lower_bound, upper_bound=upper_bound)

# Hyperparameters for the optimizer
hyperparams = {
    'reproduction': 0.3,
    'mutation': 0.4,
    'crossover': 0.4
}

# Creating GP's optimizer
p = GP(hyperparams=hyperparams)

# Creating Function's object
f = Function(pointer=benchmark.sphere)

# Finally, we can create an Opytimizer class
o = Opytimizer(space=s, optimizer=p, function=f)

# Running the optimization task
history = o.start()

print(s.best_index)

print(s.trees[s.best_index])
print(s.trees[s.best_index].position)

print(s.best_tree)
print(s.best_tree.position)

for tree in s.trees:
    print(tree)