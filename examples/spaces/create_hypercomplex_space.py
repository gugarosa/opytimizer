from opytimizer.spaces.hyper_complex import HyperComplexSpace

# We need to define the number of agents, decision variables, dimensions and iterations
n_agents = 2
n_variables = 5
n_dimensions = 4
n_iterations = 10

# Finally, we define the lower and upper bounds
# Note that they have to be the same size as n_variables
lower_bound = (0.1, 0.3, 0.5, 0.7, 0.9)
upper_bound = (0.2, 0.4, 0.6, 0.8, 1.0)

# Creating the HyperComplexSpace object
s = HyperComplexSpace(n_agents=n_agents, n_variables=n_variables,
                      n_dimensions=n_dimensions, n_iterations=n_iterations,
                      lower_bound=lower_bound, upper_bound=upper_bound)
