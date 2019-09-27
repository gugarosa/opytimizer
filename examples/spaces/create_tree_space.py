from opytimizer.spaces.tree import TreeSpace

# Firstly, we need to define the number of trees
n_trees = 1

# Also, each terminal will be rendered as an agent
n_terminals = 2

# Also, we need to define the number of decision variables
n_variables = 5

# We can also decide the number of iterations
n_iterations = 10

# Minimum depth of the trees
min_depth = 2

# Maximum depth of the trees
max_depth = 5

# List of functions nodes
func_nodes = ['SUM', 'SUB', 'MUL', 'DIV']

# Finally, we define the lower and upper bounds
# Note that they have to be the same size as n_variables
lower_bound = [0.1, 0.3, 0.5, 0.5, 0.5]
upper_bound = [0.2, 0.4, 2.0, 2.0, 2.0]

# Creating the TreeSpace object
s = TreeSpace(n_trees=n_trees, n_terminals=n_terminals, n_variables=n_variables,
              n_iterations=n_iterations, min_depth=min_depth, max_depth=max_depth,
              functions=func_nodes, lower_bound=lower_bound, upper_bound=upper_bound)

# Outputting the whole tree
print(s.trees[0])

# Outputting the tree's current position (solution)
print(f'Position: {s.trees[0].position}')

# Outputting valuable information about the tree
print(f'\nPost Order: {s.trees[0].post_order}')
print(f'\nNodes: {s.trees[0].n_nodes} | Leaves: {s.trees[0].n_leaves} | Minimum Depth: {s.trees[0].min_depth} | Maximum Depth: {s.trees[0].max_depth}')
