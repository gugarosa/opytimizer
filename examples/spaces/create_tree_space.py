from opytimizer.spaces.tree import TreeSpace

# Define the number of agents, decision variables and terminals
n_agents = 2
n_variables = 5
n_terminals = 2

# Minimum and maximum depths of the trees
min_depth = 2
max_depth = 5

# Function nodes
func_nodes = ['SUM', 'SUB', 'MUL', 'DIV']

# Also defines the corresponding lower and upper bounds
# Note that they have to be the same size as `n_variables`
lower_bound = [0.1, 0.3, 0.5, 0.7, 0.9]
upper_bound = [0.2, 0.4, 0.6, 0.8, 1.0]

# Creates the TreeSpace
s = TreeSpace(n_agents, n_variables, lower_bound, upper_bound,
              n_terminals, min_depth, max_depth, func_nodes)

# Prints out some properties
print(s.trees[0])
print(f'Position: {s.trees[0].position}')
print(f'\nPre Order: {s.trees[0].pre_order}')
print(f'\nPost Order: {s.trees[0].post_order}')
print(f'\nNodes: {s.trees[0].n_nodes} | Leaves: {s.trees[0].n_leaves} | '
      f'Minimum Depth: {s.trees[0].min_depth} | Maximum Depth: {s.trees[0].max_depth}')
