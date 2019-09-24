from opytimizer.spaces.tree import TreeSpace

# Firstly, we need to define the number of agents
n_trees = 10

# Also, we need to define the number of decision variables
n_variables = 5

# We can also decide the number of iterations
n_iterations = 10

# Minimum depth of the trees
min_depth = 2

# Maximum depth of the trees
max_depth = 5

# List of functions nodes
functions = ['SUM', 'SUB', 'MUL', 'DIV']

# List of terminal nodes
terminals = ['TERMINAL', 'CONSTANT']

# Finally, we define the lower and upper bounds
# Note that they have to be the same size as n_variables
lower_bound = [0.1, 0.3, 0.5, 0.5, 0.5]
upper_bound = [0.2, 0.4, 2.0, 2.0, 2.0]

# Creating the TreeSpace object
s = TreeSpace(n_trees=n_trees, n_variables=n_variables, n_iterations=n_iterations,
            min_depth=min_depth, max_depth=max_depth, functions=functions,
            terminals=terminals, lower_bound=lower_bound, upper_bound=upper_bound)

for tree in s.trees:
    # print(s.run_tree(tree))
    s.run_tree(tree)
