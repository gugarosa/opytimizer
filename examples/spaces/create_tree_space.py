from opytimizer.spaces.tree import TreeSpace

# Firstly, we need to define the number of agents
n_agents = 2

# Also, we need to define the number of decision variables
n_variables = 5

# We can also decide the number of iterations
n_iterations = 10

#
min_depth = 1

#
max_depth = 1

#
functions = ['SUB']

#
terminals = ['PARAM', 'CONST']

# Finally, we define the lower and upper bounds
# Note that they have to be the same size as n_variables
lower_bound = [0.1, 0.3, 0.5, 0.5, 0.5]
upper_bound = [0.2, 0.4, 2.0, 2.0, 2.0]

# Creating the TreeSpace object
s = TreeSpace(n_agents=n_agents, n_variables=n_variables, n_iterations=n_iterations, min_depth=min_depth,
              max_depth=max_depth, functions=functions, terminals=terminals,
              lower_bound=lower_bound, upper_bound=upper_bound)

print(s.constants.shape)