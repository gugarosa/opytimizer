from opytimizer.core.space import Space

# Input parameters
n_agents = 2
n_variables = 5
n_dimensions = 1
n_iterations = 5

# Bounds parameters
# Note that it has to have the same size as n_variables
lower_bound = [0.1, 0.3, 0.5, 0.5, 0.5]
upper_bound = [0.2, 0.4, 2.0, 2.0, 2.0]

# Creates a new Space
s = Space(n_agents=n_agents, n_variables=n_variables,
          n_dimensions=n_dimensions, n_iterations=n_iterations)

# Prior using the Space, you need to build it,
# so its initialized and ready for use
s.build(lower_bound=lower_bound, upper_bound=upper_bound)
