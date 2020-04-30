from opytimizer.spaces.boolean import BooleanSpace

# Firstly, we need to define the number of agents
n_agents = 2

# Also, we need to define the number of decision variables
n_variables = 5

# We can also decide the number of iterations
n_iterations = 10

# Creating the BooleanSpace object
s = BooleanSpace(n_agents=n_agents, n_variables=n_variables,
                 n_iterations=n_iterations)
