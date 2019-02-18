from opytimizer.core.space import Space

# Firstly, we need to define the number of agents
n_agents = 2

# Also, we need to define the number of decision variables and space's dimension
n_variables = 5
n_dimensions = 1

# We can also decide the number of iterations
n_iterations = 10

# Finally, we define the lower and upper bounds
# Note that they have to be the same size as n_variables
lower_bound = [0.1, 0.3, 0.5, 0.5, 0.5]
upper_bound = [0.2, 0.4, 2.0, 2.0, 2.0]

# Creating the Space object
s = Space(n_agents=n_agents, n_variables=n_variables,
          n_dimensions=n_dimensions, n_iterations=n_iterations,
          lower_bound=lower_bound, upper_bound=upper_bound)
