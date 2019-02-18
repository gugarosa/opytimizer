from opytimizer.core.space import Space

# Firstly, we need to define the number of agents and iterations to converge
n_agents = 2
n_iterations = 5

# Also, we need to define the number of decision variables and space's dimension
n_variables = 5
n_dimensions = 1

# Finally, we need the bounds
# Note that they have to be the same size as n_variables
lower_bound = [0.1, 0.3, 0.5, 0.5, 0.5]
upper_bound = [0.2, 0.4, 2.0, 2.0, 2.0]

# Creating a Space
s = Space(n_agents=n_agents, n_iterations=n_iterations,
          n_variables=n_variables, n_dimensions=n_dimensions,
          lower_bound=lower_bound, upper_bound=upper_bound)
